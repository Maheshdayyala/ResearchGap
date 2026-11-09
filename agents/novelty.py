"""Novelty Verification Agent: a prior-art search, not a score threshold.

For a candidate gap it (1) builds several targeted queries, (2) searches the four scholarly APIs plus the citation network,
(3) checks every retrieved paper against each core concept of the gap, and (4) optionally has an LLM judge the closest papers.
Verdict:
  ALREADY_ADDRESSED   a primary study (not a review) covers ALL core concepts in the gap's domain
  PARTIALLY_EXPLORED  only reviews/position papers (or LLM-judged partial matches) cover all core concepts
  NOVEL               no retrieved paper covers all core concepts together
The composite score N(g) from the design document is still computed, but only as a ranking signal, never as the decision."""
import logging
import math
import re

from agents import embeddings
from agents import lexicon as lx
from settings import settings
from utils import clip, tokens, this_year

log = logging.getLogger(__name__)


def core_concepts(gap):
    """[(kind, name, regex, label)] -- the concepts that must co-occur for a paper to 'address' the gap."""
    c, out = gap.get("components", {}), []
    for kind in ("methods", "aspects", "tasks", "datasets"):
        for name in c.get(kind, [])[:2]:
            out.append((kind, name, lx.concept_regex(kind, name), name))
    return out


class NoveltyVerificationAgent:
    def __init__(self, search_agent, llm=None):
        self.search, self.llm, self._cache = search_agent, llm, {}

    # ---------------- queries ----------------
    def queries(self, gap):
        c, topic = gap.get("components", {}), gap.get("components", {}).get("topic", "")
        method = (c.get("methods") or [""])[0]
        task = (c.get("tasks") or [""])[0]
        dataset = (c.get("datasets") or [""])[0]
        asp = (c.get("aspects") or [""])[0]
        asp_kw = lx.ASPECTS[asp][3] if asp in lx.ASPECTS else ""
        center = " ".join(x for x in (method, task, dataset, asp_kw) if x)
        qs = [f"{center} {topic}", f"{task} {asp_kw} {method}", f"{topic} {asp_kw or task or method}"]
        if not (center or topic):
            qs = [" ".join(tokens(gap["title"] + " " + gap["statement"])[:8]), gap["title"]]
        if self.llm and self.llm.available:
            r = self.llm.json(f"Research gap: {gap['statement']}\nWrite 3 diverse literature-search queries (<=8 words each) that would find any "
                              'existing paper already addressing this exact gap. JSON: {"queries":["..."]}')
            qs = [q for q in (r or {}).get("queries", []) if isinstance(q, str)][:3] + qs[:1]
        seen, out = set(), []
        for q in qs:
            q = " ".join(q.split())
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out[:4]

    # ---------------- matching ----------------
    @staticmethod
    def _domain_ok(p, topic):
        """Require more than one topical anchor; one shared acronym/word is not domain evidence."""
        text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        raw = re.findall(r"[a-z0-9]+", (topic or "").lower())
        stop = {"in", "on", "for", "and", "with", "under", "using", "of", "to", "the", "a", "an", "from", "based"}
        anchors = [t for t in raw if t not in stop and len(t) > 1]
        if not anchors:
            return True
        # Disambiguate the common AAI acronym collision unless the query explicitly means animal-assisted intervention.
        q = topic.lower()
        if re.search(r"\\baai\\b", q) and not ("animal assisted" in q or "animal-assisted" in q):
            if "animal-assisted" in text or "animal assisted" in text:
                return False
        hits = sum(1 for t in set(anchors) if re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", text))
        required = max(2, min(4, math.ceil(0.35 * len(set(anchors)))))
        return hits >= required

    def _coverage(self, p, core, topic):
        text = f"{p['title']}. {p['abstract']}"
        hit = [name for _, name, rx, _ in core if rx.search(text)]
        return hit, (len(hit) == len(core) and bool(core) and self._domain_ok(p, topic))

    # ---------------- main ----------------
    def verify(self, gap, corpus, expand=True):
        cy, c = this_year(), gap.get("components", {})
        topic, core = c.get("topic", ""), core_concepts(gap)
        pool, search_log, notes = {p["pid"]: p for p in corpus}, [], []
        if expand:
            for q in self.queries(gap):
                try:
                    if q not in self._cache:
                        self._cache[q] = self.search.fetch_ex(q, per_source=settings.PRIOR_ART_PER_SOURCE)
                    papers, status = self._cache[q]
                    search_log.append({"query": q, "results": len(papers), "sources": status})
                    for p in papers:
                        pool.setdefault(p["pid"], p)
                except Exception as e:
                    search_log.append({"query": q, "error": type(e).__name__})
            try:   # citation-network expansion: who cites the closest prior work?
                close = sorted(pool.values(), key=lambda p: -(p.get("relevance") or 0))[:2]
                added = 0
                for p in close:
                    for cp in self.search.citing_papers(p, 25):
                        if cp["pid"] not in pool:
                            pool[cp["pid"]], added = cp, added + 1
                search_log.append({"query": "citation network of closest papers", "results": added})
            except Exception as e:
                notes.append(f"citation expansion unavailable ({type(e).__name__})")
        papers = [p for p in pool.values() if p.get("abstract") and self._domain_ok(p, topic)]
        if not papers:
            return self._empty(gap, search_log, notes + ["Retrieved records did not match enough topic-specific anchors; possible acronym/domain drift."])

        gtext = f"{gap['title']}. {gap['statement']}"
        sims = embeddings.embed([f"{p['title']}. {p['abstract'][:600]}" for p in papers]) @ embeddings.embed([gtext])[0]
        rows = []
        for p, s in zip(papers, sims):
            hit, full = self._coverage(p, core, topic)
            rev = lx.is_review(p["title"], p["abstract"])
            rows.append({"p": p, "sim": float(s), "hit": hit, "full": full, "review": rev})
        rows.sort(key=lambda r: (-int(r["full"]), -len(r["hit"]), -r["sim"]))

        full_study = [r for r in rows if r["full"] and not r["review"]]
        full_rev = [r for r in rows if r["full"] and r["review"]]
        judged = []
        if self.llm and self.llm.available:
            for r in rows[:4]:
                j = self.llm.json(f'Research gap: {gap["statement"]}\nPaper: {r["p"]["title"]}. {r["p"]["abstract"][:1000]}\n'
                                  'Does this paper itself study the gap (same concepts studied together)? JSON: {"verdict":"addresses|partial|no","reason":"..."}')
                if j and j.get("verdict") in ("addresses", "partial", "no"):
                    r["llm"], r["reason"] = j["verdict"], str(j.get("reason", ""))[:200]
                    judged.append(r)
            for r in judged:   # LLM overrides lexical matching in both directions
                if r["llm"] == "no" and r in full_study:
                    full_study.remove(r)
                elif r["llm"] == "addresses" and not r["review"] and r not in full_study:
                    full_study.append(r)
                elif r["llm"] == "partial" and r not in full_rev and r not in full_study:
                    full_rev.append(r)
        nsrc = len({s for q in search_log for s, v in (q.get("sources") or {}).items() if str(v).startswith("ok")})
        if full_study:
            verdict = "ALREADY_ADDRESSED"
        elif full_rev:
            verdict = "PARTIALLY_EXPLORED"
        elif len(papers) < 8 or (expand and nsrc == 0 and len(papers) < 20):
            verdict = "INSUFFICIENT_EVIDENCE"
        else:
            verdict = "NOVEL"

        conf = "HIGH" if (len(papers) >= 80 and nsrc >= 3 and judged) else "MEDIUM" if (len(papers) >= 40 and nsrc >= 2) else "LOW"
        conf_why = f"{len(papers)} papers compared, {nsrc} scholarly source(s) responded, {'LLM-judged' if judged else 'rule-based matching only'}."
        top = (full_study or full_rev or rows)[:5]
        mk = lambda r: {"pid": r["p"]["pid"], "title": r["p"]["title"], "year": r["p"].get("year"), "doi": r["p"].get("doi"), "url": r["p"].get("url"),
                        "similarity": round(r["sim"], 3), "matched": r["hit"], "type": "review" if r["review"] else "study", "llm": r.get("llm"), "reason": r.get("reason", "")}
        return {"verdict": verdict, "accepted": verdict == "NOVEL", "confidence": conf, "confidence_reason": conf_why,
                "core_concepts": [lbl for *_, lbl in core], "pool_size": len(papers), "search_log": search_log,
                "closest": [mk(r) for r in top], "full_match_studies": len(full_study), "full_match_reviews": len(full_rev),
                "score": self._signal(rows, gap, cy), "notes": notes + (["No scholarly API responded; verdict is based on the retrieved corpus only."] if not nsrc and expand else []),
                "threshold": None, "supporting": gap.get("evidence", []),
                "explanation": self._explain(verdict, core, full_study, full_rev, len(papers))}

    @staticmethod
    def _explain(verdict, core, fs, fr, n):
        names = " + ".join(lbl for *_, lbl in core) or "the gap's concepts"
        if verdict == "INSUFFICIENT_EVIDENCE":
            return f"Only {n} topic-matched papers were available. There is not enough relevant prior-art coverage to call this gap novel or already addressed; refine the query or add sources."
        if verdict == "NOVEL":
            return f"Searched {n} topic-matched papers: none studies {names} together. Absence of evidence in these databases is not proof; confirm manually before committing."
        if verdict == "PARTIALLY_EXPLORED":
            return f"Only review/position papers ({len(fr)}) discuss {names} together; no primary study found. A fresh empirical study is still plausible."
        return f"{len(fs)} primary stud{'y' if len(fs) == 1 else 'ies'} already study {names} together (see closest work). Choose a narrower angle."

    @staticmethod
    def _signal(rows, gap, cy):
        """Design-document composite N(g) = .40 Nsem + .25 Ncov + .20 Nrec + .15 Nevi (ranking signal only)."""
        maxsim = max((r["sim"] for r in rows), default=0)
        nfull = sum(r["full"] for r in rows)
        close = [r for r in rows if r["sim"] >= 0.5 * settings.DIRECT_SIM + 0.1]
        latest = max((r["p"].get("year") or 0 for r in close), default=0)
        nrec = 1.0 if not latest else clip((cy - latest) / 5)
        nevi = clip(len(gap.get("evidence", [])) / 4 + (0.25 if gap["type"] == "CONTRADICTION" else 0))
        return round(float(settings.W_SEM * clip(1 - maxsim) + settings.W_COV * clip(1 - nfull / 3) + settings.W_REC * nrec + settings.W_EVI * nevi), 3)

    @staticmethod
    def _empty(gap, log_, notes):
        return {"verdict": "INSUFFICIENT_EVIDENCE", "accepted": False, "confidence": "LOW", "confidence_reason": "no literature could be retrieved", "core_concepts": [],
                "pool_size": 0, "search_log": log_, "closest": [], "full_match_studies": 0, "full_match_reviews": 0, "score": 0.0,
                "notes": notes + ["No literature available to compare against; not accepted."], "threshold": None, "supporting": gap.get("evidence", []),
                "explanation": "No papers could be retrieved for the prior-art search."}
