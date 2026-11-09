"""ResearchGapAI – Flask REST backend + web interface."""
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_wtf.csrf import CSRFProtect, generate_csrf
from werkzeug.exceptions import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.middleware.proxy_fix import ProxyFix

import services
from agents import embeddings, get_agents
from models import ChatMessage, GapReport, Paper, Search, User, db
from settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)
csrf, limiter, login_manager = CSRFProtect(), Limiter(key_func=get_remote_address, default_limits=["300 per hour"], storage_uri="memory://"), LoginManager()
executor = ThreadPoolExecutor(max_workers=2)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def clean_query(q):
    q = re.sub(r"[\x00-\x1f<>]", " ", (q or "")).strip()
    q = re.sub(r"\s+", " ", q)
    if not 3 <= len(q) <= 200:
        abort(400, "Query must be between 3 and 200 characters.")
    return q


def create_app(test_config=None):
    app = Flask(__name__)
    # Trust one reverse proxy for scheme/host headers in deployment; localhost remains HTTP.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.config.update(SECRET_KEY=settings.SECRET_KEY, SQLALCHEMY_DATABASE_URI=settings.DATABASE_URL,
                      SQLALCHEMY_TRACK_MODIFICATIONS=False, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=settings.COOKIE_SECURE,
                      WTF_CSRF_TIME_LIMIT=None,
                      MAX_CONTENT_LENGTH=1_000_000)
    if test_config:
        app.config.update(test_config)
    db.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "login"
    with app.app_context():
        db.create_all()

    wants_json = lambda: request.is_json or request.accept_mimetypes.best == "application/json" or request.headers.get("X-Requested-With")

    @login_manager.user_loader
    def load_user(uid):
        return db.session.get(User, int(uid))

    @login_manager.unauthorized_handler
    def unauthorized():
        return (jsonify(error="Authentication required"), 401) if wants_json() else redirect(url_for("login", next=request.path))

    @app.errorhandler(HTTPException)
    def http_err(e):
        return jsonify(error=e.description, code=e.code), e.code

    @app.errorhandler(Exception)
    def unexpected_err(e):
        log.exception("Unhandled request error on %s", request.path)
        return jsonify(error="The server could not complete this request. The error was logged for diagnosis.", code=500), 500

    @app.after_request
    def headers(r):
        r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "same-origin"})
        if request.is_secure:
            r.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return r

    def own(search_id):
        s = db.session.get(Search, search_id) or abort(404, "Search not found")
        if s.user_id != current_user.id and not current_user.is_admin:
            abort(403, "Not your search")
        return s

    def body():
        return request.get_json(silent=True) or {}

    psum = lambda p: {k: p[k] for k in ("db_id", "pid", "title", "authors", "year", "venue", "doi", "url", "citations", "sources", "rank_score")} | {"abstract": p["abstract"][:700], "structured": p.get("structured")}

    # ------------------------------------------------------------ pages / auth
    @app.get("/")
    @login_required
    def index():
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    @limiter.limit("10 per hour", methods=["POST"])
    def register():
        if request.method == "POST":
            email, pw = request.form.get("email", "").strip().lower(), request.form.get("password", "")
            if not EMAIL_RE.match(email) or len(pw) < 8:
                return render_template("auth.html", mode="register", error="Enter a valid email and a password of at least 8 characters."), 400
            if User.query.filter_by(email=email).first():
                return render_template("auth.html", mode="register", error="That email is already registered."), 400
            u = User(email=email, is_admin=bool(settings.ADMIN_EMAIL and email == settings.ADMIN_EMAIL))
            u.set_password(pw)
            db.session.add(u)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                return render_template("auth.html", mode="register", error="That email is already registered."), 400
            login_user(u)
            return redirect(url_for("index"))
        return render_template("auth.html", mode="register")

    @app.route("/login", methods=["GET", "POST"])
    @limiter.limit("20 per hour", methods=["POST"])
    def login():
        if request.method == "POST":
            u = User.query.filter_by(email=request.form.get("email", "").strip().lower()).first()
            if u and u.check_password(request.form.get("password", "")):
                login_user(u)
                nxt = request.args.get("next", "")
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("index"))
            return render_template("auth.html", mode="login", error="Email or password is incorrect."), 401
        return render_template("auth.html", mode="login")

    @app.post("/logout")
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.get("/history")
    @login_required
    def history_page():
        return render_template("history.html")

    @app.get("/api/csrf-token")
    @login_required
    def csrf_token():
        return jsonify(csrf_token=generate_csrf())

    @app.get("/evaluation")
    @login_required
    def evaluation_page():
        return render_template("evaluation.html")

    @app.get("/api/evaluation")
    @login_required
    def evaluation_history():
        rows = db.session.query(Search).filter_by(user_id=current_user.id).filter(Search.meta.isnot(None)).order_by(Search.created_at.asc()).all()
        return jsonify([{"search_id": s.id, "query": s.query, "status": s.status, "created": s.created_at.isoformat(),
                         "agents": (s.meta or {}).get("agent_evaluation", [])} for s in rows])

    @app.get("/api/history")
    @login_required
    def history():
        q = db.session.query(Search)
        if not (current_user.is_admin and request.args.get("all")):
            q = q.filter_by(user_id=current_user.id)
        rows = q.order_by(Search.created_at.desc()).limit(100).all()
        return jsonify([{"search_id": s.id, "query": s.query, "status": s.status, "created": s.created_at.isoformat(),
                         "report_id": s.report.id if s.report else None} for s in rows])

    @app.get("/health")
    def health():
        ag = get_agents()
        db_backend = "postgresql" if str(app.config["SQLALCHEMY_DATABASE_URI"]).startswith(("postgresql://", "postgresql+")) else "sqlite"
        db_ok = False
        try:
            db.session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db.session.rollback()
        return jsonify(status="ok" if db_ok else "degraded", database={"backend": db_backend, "connected": db_ok},
                       neo4j={"configured": bool(settings.NEO4J_URI), "connected": bool(ag.kg_builder.store.ok)},
                       llm=ag.llm.provider, embeddings=embeddings.backend())

    @app.get("/admin/stats")
    @login_required
    def admin_stats():
        if not current_user.is_admin:
            abort(403, "Admin only")
        return jsonify(users=User.query.count(), searches=db.session.query(Search).count(), papers=Paper.query.count(), reports=GapReport.query.count(),
                       failed=db.session.query(Search).filter_by(status="failed").count())

    # ------------------------------------------------------------ REST API (Table 3)
    @app.post("/search")
    @limiter.limit("30 per hour")
    @login_required
    def search():
        q = clean_query(body().get("query"))
        s = Search(user_id=current_user.id, query=q, status="running", stage="search")
        db.session.add(s)
        db.session.commit()
        try:
            papers, status = get_agents().search.fetch_ex(q)
            services.save_papers(s, papers)
            s.status, s.meta = "retrieved", {"source_status": status}
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            s = db.session.get(Search, s.id)
            s.status, s.stage, s.error, s.message = "failed", "search", f"{type(e).__name__}: {e}"[:1000], "Literature search failed"
            db.session.commit()
            raise
        return jsonify(search_id=s.id, source_status=status, count=len(papers), papers=[psum(p) for p in services.load_papers(s)])

    @app.get("/papers/<int:pid>")
    @login_required
    def paper(pid):
        p = db.session.get(Paper, pid) or abort(404, "Paper not found")
        own(p.search_id)
        return jsonify(p.to_dict())

    @app.post("/understand")
    @login_required
    def understand():
        d, ag = body(), get_agents()
        if d.get("paper_id"):
            p = db.session.get(Paper, int(d["paper_id"])) or abort(404, "Paper not found")
            own(p.search_id)
            p.structured = ag.understand.extract(p.to_dict())
            db.session.commit()
            return jsonify(paper_id=p.id, structured=p.structured)
        if d.get("abstract"):
            return jsonify(structured=ag.understand.extract({"title": str(d.get("title", ""))[:300], "abstract": str(d["abstract"])[:6000]}))
        if d.get("search_id"):
            s = own(int(d["search_id"]))
            Paper.query.filter_by(search_id=s.id).update({"structured": None})
            papers = services.ensure_understood(s)
            return jsonify(search_id=s.id, count=len(papers), papers=[{"db_id": p["db_id"], "title": p["title"], "structured": p["structured"]} for p in papers])
        abort(400, "Provide paper_id, search_id, or title+abstract.")

    @app.post("/gap-detect")
    @login_required
    def gap_detect():
        d, ag = body(), get_agents()
        s = own(int(d.get("search_id") or abort(400, "search_id required")))
        papers = services.ensure_understood(s)
        if not papers:
            abort(400, "Run /search first.")
        kg = ag.kg_builder.build(s.id, papers, query=s.query)
        cands = ag.gap.detect(kg, int(d.get("round", 0)))
        r = services.get_report(s, current_user.id)
        r.candidates, r.status = cands, "candidates"
        db.session.commit()
        return jsonify(search_id=s.id, report_id=r.id, graph=kg.stats(), topics=kg.topics, candidates=cands)

    @app.post("/verify-novelty")
    @login_required
    def verify_novelty():
        d, ag = body(), get_agents()
        s = own(int(d.get("search_id") or abort(400, "search_id required")))
        r = services.get_report(s, current_user.id)
        gap = d.get("gap")
        if not gap and d.get("gap_id"):
            gap = next((c for c in r.candidates or [] if c["id"] == d["gap_id"]), None)
        if not gap or "statement" not in gap:
            abort(400, "Provide gap (object with statement) or a gap_id from /gap-detect.")
        gap.setdefault("type", "CUSTOM"); gap.setdefault("title", gap["statement"][:80]); gap.setdefault("components", {}); gap.setdefault("evidence", []); gap.setdefault("id", "custom")
        res = ag.novelty.verify(gap, services.ensure_understood(s))
        cands = [c for c in (r.candidates or []) if c.get("id") != gap["id"]] + [{**gap, "novelty": res}]
        r.candidates = cands
        if res["accepted"] or (res["verdict"] == "PARTIALLY_EXPLORED" and not r.verified_gap):
            r.verified_gap, r.novelty = gap, res
        db.session.commit()
        return jsonify(search_id=s.id, gap_id=gap["id"], novelty=res)

    @app.post("/experiment-plan")
    @login_required
    def experiment_plan():
        d, ag = body(), get_agents()
        s = own(int(d.get("search_id") or abort(400, "search_id required")))
        r = services.get_report(s, current_user.id)
        gap = d.get("gap") or r.verified_gap or abort(400, "No verified gap yet; call /verify-novelty first.")
        r.plan = ag.planner.plan(gap, services.ensure_understood(s), r.novelty)
        # Planning a candidate is not the same as verifying its novelty.
        r.plan["provisional"] = not bool(r.novelty and r.novelty.get("verdict") in ("NOVEL", "PARTIALLY_EXPLORED"))
        db.session.commit()
        return jsonify(search_id=s.id, plan=r.plan)

    @app.get("/report/<int:rid>")
    @login_required
    def report(rid):
        r = db.session.get(GapReport, rid) or abort(404, "Report not found")
        s = own(r.search_id)
        md = r.report_md or services.render_report(s, r)
        fmt = request.args.get("format", "json")
        if fmt == "md":
            return md, 200, {"Content-Type": "text/markdown; charset=utf-8", "Content-Disposition": f"attachment; filename=report_{r.id}.md"}
        html = get_agents().report.to_html(md)
        if fmt == "html":
            return html
        papers = services.load_papers(s)
        kg = get_agents().kg_builder.build(s.id, papers, persist=False, query=s.query) if papers else None
        graph = None
        if kg:
            graph = {"stats": kg.stats(), "topics": kg.topics,
                     "nodes": [{"id": key, **{k: v for k, v in attrs.items() if k in ("kind", "name", "label", "title", "year", "pid")}}
                               for key, attrs in kg.g.nodes(data=True)],
                     "edges": [{"source": u, "target": v, "relation": attrs.get("rel"), "weight": attrs.get("weight", 1)}
                               for u, v, attrs in kg.g.edges(data=True)]}
        return jsonify(id=r.id, search_id=s.id, query=s.query, status=r.status, meta=s.meta, verified_gap=r.verified_gap, novelty=r.novelty,
                       plan=r.plan, candidates=r.candidates, papers=[psum(p) for p in papers], graph=graph, report_md=md, report_html=html, agent_evaluation=(s.meta or {}).get("agent_evaluation", []))

    # ------------------------------------------------------------ analysis-scoped research assistant
    @app.get("/api/analysis/<int:sid>/chat")
    @login_required
    def get_analysis_chat(sid):
        own(sid)
        rows = ChatMessage.query.filter_by(search_id=sid, user_id=current_user.id).order_by(ChatMessage.created_at.asc()).all()
        return jsonify([{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in rows])

    @app.post("/api/analysis/<int:sid>/chat")
    @login_required
    @limiter.limit("30 per hour")
    def post_analysis_chat(sid):
        s = own(sid)
        question = str(body().get("message", "")).strip()[:2000]
        if len(question) < 2:
            abort(400, "Enter a question about this analysis.")
        rep = GapReport.query.filter_by(search_id=s.id, user_id=current_user.id).first()
        if not rep:
            abort(404, "Analysis report is not ready yet.")
        ag = get_agents()
        history = ChatMessage.query.filter_by(search_id=s.id, user_id=current_user.id).order_by(ChatMessage.created_at.desc()).limit(20).all()[::-1]
        db.session.add(ChatMessage(search_id=s.id, user_id=current_user.id, role="user", content=question))
        db.session.commit()
        papers = services.load_papers(s)[:35]
        paper_context = "\\n".join(f"- [{p.get('pid')}] {p['title']} ({p.get('year') or 'n.d.'}): {p.get('abstract','')[:650]}" for p in papers)
        graph_context = (s.meta or {}).get("graph") or {}
        context = f"Research query: {s.query}\\nVerified/potential gap: {rep.verified_gap or 'No gap was verified.'}\\nNovelty assessment: {rep.novelty or 'Not available.'}\\nExperiment plan: {rep.plan or 'Not available.'}\\nCandidate gaps: {(rep.candidates or [])[:15]}\\nKnowledge graph summary: {graph_context}\\nRetrieved papers and citation IDs:\\n{paper_context}\\nReport excerpt:\\n{(rep.report_md or '')[:9000]}"
        prior = "\\n".join(f"{m.role}: {m.content}" for m in history)
        try:
            if not ag.llm.available:
                raise RuntimeError("language model is not configured")
            answer = ag.llm.complete(question, system=("You are ResearchGapAI's research assistant. Answer in the context of the user's saved analysis. "
                "Use only supplied analysis facts for claims about its papers and outputs. Clearly say when evidence is missing; "
                "do not invent citations, results, datasets, or novelty claims. Distinguish general suggestions from retrieved evidence.\n\n"
                + context + "\n\nRecent conversation:\n" + prior), max_tokens=1200)
        except Exception as e:
            log.warning("Analysis assistant is using saved-evidence fallback (%s)", type(e).__name__)
            verdict = (rep.novelty or {}).get("verdict")
            labels = {"NOVEL": "Potentially unexplored", "PARTIALLY_EXPLORED": "Partially addressed",
                      "ALREADY_ADDRESSED": "Already addressed", "INSUFFICIENT_EVIDENCE": "Insufficient evidence"}
            if rep.verified_gap:
                answer = (f"The saved analysis proposes “{rep.verified_gap.get('title', 'a research direction')}.” "
                          f"{rep.verified_gap.get('statement', '')}\n\nPrior-art result: {labels.get(verdict, verdict or 'not verified')}; "
                          f"confidence: {(rep.novelty or {}).get('confidence', 'not recorded')}. "
                          f"{(rep.novelty or {}).get('explanation', 'Review the supporting papers and search coverage before making a novelty claim.')}")
            else:
                answer = "This saved analysis did not verify a recommended gap. Open All candidates to review the candidates and why they were rejected or marked uncertain."
            closest = ((rep.novelty or {}).get("closest") or [])[:3]
            if closest:
                answer += "\n\nClosest retrieved papers: " + "; ".join(
                    f"{p.get('title')} ({p.get('year') or 'n.d.'})" for p in closest) + "."
            answer += "\n\nThis is an evidence summary from your saved analysis; the language-model service is temporarily unavailable or rate-limited."
        db.session.add(ChatMessage(search_id=s.id, user_id=current_user.id, role="assistant", content=answer[:12000]))
        db.session.commit()
        return jsonify(role="assistant", content=answer[:12000])

    # ------------------------------------------------------------ full pipeline (async) + status
    @app.post("/pipeline")
    @limiter.limit("10 per hour")
    @login_required
    def pipeline():
        q = clean_query(body().get("query"))
        s = Search(user_id=current_user.id, query=q, status="queued", stage="queued", message="Queued")
        db.session.add(s)
        db.session.commit()
        r = services.get_report(s, current_user.id)
        executor.submit(services.run_pipeline, app, s.id, current_user.id)
        return jsonify(search_id=s.id, report_id=r.id), 202

    @app.get("/status/<int:sid>")
    @limiter.exempt
    @login_required
    def status(sid):
        s = own(sid)
        return jsonify(status=s.status, stage=s.stage, progress=s.progress, message=s.message, error=s.error,
                       report_id=s.report.id if s.report else None)

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=False, threaded=True)
