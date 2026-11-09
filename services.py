"""DB <-> agent glue shared by the REST endpoints and the background pipeline."""
import logging

from agents import get_agents
from models import Citation, GapReport, Paper, Search, db
from workflow import run_workflow

log = logging.getLogger(__name__)


def save_papers(search, papers):
    Paper.query.filter_by(search_id=search.id).delete()
    for p in papers:
        db.session.add(Paper(search_id=search.id, pid=p["pid"], title=p["title"], authors=p.get("authors"), year=p.get("year"),
                             abstract=p.get("abstract"), venue=(p.get("venue") or "")[:300], doi=p.get("doi"), url=(p.get("url") or "")[:500],
                             citations_count=p.get("citations", 0), rank_score=p.get("rank_score", 0), sources=p.get("sources"),
                             external_ids=p.get("external_ids"), references=p.get("references"), structured=p.get("structured")))
    Citation.query.filter_by(search_id=search.id).delete()
    ids = {(p.get("external_ids") or {}).get("openalex"): p["pid"] for p in papers}
    for p in papers:
        for r in p.get("references", []):
            if ids.get(r) and ids[r] != p["pid"]:
                db.session.add(Citation(search_id=search.id, citing_pid=p["pid"], cited_pid=ids[r]))
    db.session.commit()


def load_papers(search):
    return [p.to_dict() for p in Paper.query.filter_by(search_id=search.id).order_by(Paper.rank_score.desc()).all()]


def ensure_understood(search):
    ag, rows = get_agents(), Paper.query.filter_by(search_id=search.id).all()
    todo = [r for r in rows if not r.structured]
    if todo:
        for r, st in zip(todo, ag.understand.extract_many([r.to_dict() for r in todo])):
            r.structured = st
        db.session.commit()
    return load_papers(search)


def get_report(search, user_id):
    r = GapReport.query.filter_by(search_id=search.id).first()
    if not r:
        r = GapReport(search_id=search.id, user_id=user_id, candidates=[], status="draft")
        db.session.add(r)
        db.session.commit()
    return r


def render_report(search, report):
    """(Re)generate report markdown from whatever stages have been completed for this search."""
    ag = get_agents()
    papers = load_papers(search)
    kg = ag.kg_builder.build(search.id, papers, persist=False, query=search.query) if papers else None
    ev = [c for c in (report.candidates or []) if c.get("novelty")]
    md = ag.report.generate({"query": search.query, "papers": papers, "topics": kg.topics if kg else [],
                             "candidates": sorted(ev, key=lambda c: -c["novelty"]["score"]) or (report.candidates or []),
                             "verified_gap": report.verified_gap, "novelty": report.novelty, "plan": report.plan})
    report.report_md = md
    db.session.commit()
    return md


def run_pipeline(app, search_id, user_id):
    """Background thread entry point: runs the full LangGraph workflow and persists every artefact."""
    with app.app_context():
        search = db.session.get(Search, search_id)
        ag = get_agents()

        def emit(stage, pct, msg):
            s = db.session.get(Search, search_id)
            s.status, s.stage, s.progress, s.message = "running", stage, pct, msg[:390]
            db.session.commit()

        try:
            final = run_workflow(ag, search_id, search.query, emit)
            from agents import embeddings
            save_papers(search, final.get("papers", []))
            rep = get_report(search, user_id)
            rep.candidates = final.get("evaluated", [])
            rep.verified_gap, rep.novelty, rep.plan = final.get("verified_gap"), final.get("novelty"), final.get("plan")
            rep.report_md, rep.status = final.get("report_md", ""), final.get("status", "done")
            kg = final.get("kg")
            paper_count = len(final.get("papers", []))
            structured_count = sum(bool(p.get("structured")) for p in final.get("papers", []))
            evaluated = final.get("evaluated", [])
            verdict_counts = {}
            for item in evaluated:
                verdict = (item.get("novelty") or {}).get("verdict", "NOT_CHECKED")
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            graph_stats = kg.stats() if kg else {}
            node_total = sum((graph_stats.get("nodes") or {}).values()) if isinstance(graph_stats.get("nodes"), dict) else 0
            edge_total = sum((graph_stats.get("edges") or {}).values()) if isinstance(graph_stats.get("edges"), dict) else 0
            agent_evaluation = [
                {"agent": "Literature Search", "status": "completed" if paper_count else "no results",
                 "output": f"{paper_count} papers retrieved", "measure": paper_count, "unit": "papers", "note": f"Source status: {final.get('source_status') or {}}"},
                {"agent": "Paper Understanding", "status": "completed" if structured_count else "no structured papers",
                 "output": f"{structured_count}/{paper_count} papers structured", "measure": structured_count, "unit": "papers structured", "note": "Methods, tasks, datasets and limitations are extracted when present in source metadata."},
                {"agent": "Knowledge Graph", "status": "completed" if kg else "not built",
                 "output": f"{node_total} nodes; {edge_total} edges" if kg else "No graph output", "measure": node_total + edge_total, "unit": "graph elements", "note": f"Backend: {graph_stats.get('backend', 'unavailable')}"},
                {"agent": "Research Gap Detection", "status": "completed" if evaluated else "no candidates",
                 "output": f"{len(evaluated)} candidates sent to prior-art verification", "measure": len(evaluated), "unit": "candidates", "note": "Candidates retain their supporting paper evidence."},
                {"agent": "Novelty Verification", "status": "completed" if evaluated else "not run",
                 "output": f"{len(evaluated)} candidates checked; verdicts: {verdict_counts}", "measure": len(evaluated), "unit": "prior-art checks", "note": "A database search cannot prove global novelty."},
                {"agent": "Experiment Planning", "status": "completed" if final.get('plan') else "not generated",
                 "output": "Experiment plan generated" if final.get('plan') else "No plan generated", "measure": len((final.get('plan') or {}).get('methodology', [])), "unit": "plan steps", "note": "Plan suggestions require domain and dataset suitability review."}
            ]
            search.meta = {"source_status": final.get("source_status"), "embedding_backend": embeddings.backend(),
                           "llm": ag.llm.provider if ag.llm.available else "heuristic", "graph": graph_stats or None,
                           "corpus": final.get("corpus_stats"), "refinements": final.get("retries", 0), "detect_rounds": final.get("detect_round", 0),
                           "agent_evaluation": agent_evaluation}
            search.status, search.stage, search.progress = rep.status, "done", 100
            search.message = "Report ready" if rep.status == "done" else "No literature found"
            rep.report_md = ag.report.generate({"query": search.query, "papers": final.get("papers", []),
                "topics": kg.topics if kg else [], "candidates": evaluated, "verified_gap": rep.verified_gap,
                "novelty": rep.novelty, "plan": rep.plan, "corpus_stats": final.get("corpus_stats"),
                "agent_evaluation": agent_evaluation})
            db.session.commit()
        except Exception as e:  # noqa
            log.exception("pipeline failed")
            db.session.rollback()
            s = db.session.get(Search, search_id)
            s.status, s.error, s.message = "failed", f"{type(e).__name__}: {e}"[:1000], "Pipeline failed"
            s.meta = {**(s.meta or {}), "agent_evaluation": [{"agent": "Pipeline", "status": "failed", "output": s.error, "measure": 0, "unit": "completed outputs", "note": f"Failure occurred during stage: {s.stage}."}]}
            db.session.commit()
