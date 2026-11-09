"""Research Gap Detection Agent.
A gap is a SPECIFIC, checkable statement: two concepts that are each well studied in the corpus but never studied together
(method x aspect, task x aspect, dataset x aspect, method x task, method x dataset), a concern that reviews keep raising but
primary studies never test, a real contradiction between primary studies, or a concrete limitation repeated across studies."""
import hashlib
import re
import math
from collections import defaultdict

from agents import embeddings
from agents import lexicon as lx
from utils import clip, tokens

W = {"CONCEPT_GAP": 1.0, "UNTESTED_CONCERN": 0.95, "CONTRADICTION": 0.9, "REPEATED_LIMITATION": 0.8}
PAIRS = [("methods", "aspects", 1.0), ("tasks", "aspects", 0.9), ("datasets", "aspects", 0.85), ("methods", "tasks", 0.8), ("methods", "datasets", 0.7)]
ASP = lambda a: lx.ASPECTS[a]


class ResearchGapDetectionAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def detect(self, kg, round_=0, exclude=(), limit=12):
        """round_>0 relaxes thresholds (fewer papers required, allow one existing co-occurrence)."""
        topic = (kg.query or "this domain").strip()
        idx = self._index(kg)
        cands = (self._concept_gaps(kg, idx, topic, round_) + self._untested(kg, idx, topic, round_) +
                 self._contradictions(kg, topic) + self._limitations(kg, topic, round_))
        seen, out = set(), []
        for c in sorted(cands, key=lambda c: -c["score"]):
            c["id"] = "g" + hashlib.sha1((c["title"] + c["statement"]).encode()).hexdigest()[:8]
            if c["id"] in seen or c["statement"] in exclude or c["title"] in exclude:
                continue
            seen.add(c["id"])
            out.append(c)
        out = out[:limit]
        self._polish(out, topic)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _index(kg):
        idx = {f: defaultdict(lambda: {"all": set(), "study": set(), "review": set()}) for f in ("methods", "aspects", "tasks", "datasets")}
        topic = (kg.query or "").lower()
        topic_terms = [t for t in re.findall(r"[a-z0-9]+", topic.lower()) if t not in {"in", "on", "for", "and", "with", "under", "using", "of", "to", "the", "a", "an"} and len(t) > 1]
        for pid, p in kg.papers.items():
            text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            if len(set(topic_terms)) >= 2:
                hits = sum(1 for term in set(topic_terms) if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text))
                if hits < max(2, min(4, math.ceil(0.35 * len(set(topic_terms))))):
                    continue
            if "aai" in topic_terms and "animal assisted" not in topic and ("animal-assisted" in text or "animal assisted" in text):
                continue
            st = p.get("structured") or {}
            kind = "review" if st.get("paper_type") == "review" else "study"
            for f in idx:
                for name in st.get(f, []):
                    # Don't turn a neighboring corpus task into the user's gap if the task is outside their query.
                    if f == "tasks" and not lx.concept_regex("tasks", name).search(topic):
                        continue
                    idx[f][name]["all"].add(pid)
                    idx[f][name][kind].add(pid)
        return idx

    @staticmethod
    def _ev(kg, items):
        out = []
        for pid, role in items:
            p = kg.papers[pid]
            st = p.get("structured") or {}
            out.append({"pid": pid, "title": p["title"], "year": p.get("year"), "doi": p.get("doi"), "role": role,
                        "snippet": (st.get("goal") or p["abstract"][:200])[:260]})
        return out

    @staticmethod
    def _comp(topic, **kw):
        c = {"methods": [], "aspects": [], "tasks": [], "datasets": [], "topic": topic}
        c.update(kw)
        return c

    def _concept_gaps(self, kg, idx, topic, rnd):
        out, min_a, max_co = [], max(1, 2 - (rnd >= 2)), (0 if rnd < 2 else 1)
        query_tasks, query_aspects = set(lx.tasks_in(topic)), set(lx.aspects_in(topic))
        for fa, fb, w in PAIRS:
            rows = []
            if (fa, fb) == ("methods", "datasets") and kg.unexplored_pairs:
                for r in kg.unexplored_pairs(min_use=min_a, limit=10):
                    if r["method"] in lx.GENERIC_METHODS:
                        continue
                    rows.append((r["method"], r["dataset"], r["mc"], r["dc"], 0))
            else:
                for a, ia in idx[fa].items():
                    if len(ia["study"]) < min_a or a in lx.GENERIC_METHODS or (fa == "tasks" and a in topic.lower()):
                        continue
                    for b, ib in idx[fb].items():
                        nb = len(ib["all"]) if fb == "aspects" else len(ib["study"])
                        co = len(ia["study"] & ib["study"])
                        if nb >= 2 and co <= max_co:
                            rows.append((a, b, len(ia["study"]), nb, co))
            rows.sort(key=lambda r: -math.log1p(r[2]) * math.log1p(r[3]))
            for a, b, na, nb, co in rows[:3]:
                raw = clip(math.log1p(na) * math.log1p(nb) / (math.log(8) ** 2)) * w
                # Rank gaps about concepts stated in the user's query above incidental corpus pairs.
                if fa == "tasks" and a in query_tasks:
                    raw *= 1.5
                elif fa == "methods" and lx.concept_regex("methods", a).search(topic):
                    raw *= 1.25
                if fb == "aspects" and b in query_aspects:
                    raw *= 1.5
                out.append(self._make_concept(kg, idx, topic, fa, fb, a, b, na, nb, co, raw))
        return out

    def _make_concept(self, kg, idx, topic, fa, fb, a, b, na, nb, co, raw):
        part = "only once" if co else "never"
        if fb == "aspects":
            d, why = ASP(b)[1], ASP(b)[2]
            if fa == "methods":
                title, miss = f"{a.capitalize()} and {b}", f"Systematic evaluation of {d} when using {a} for {topic}."
                stmt = f"No retrieved primary study examines {a} with respect to {b} in {topic} ({part} together), although {a} is used in {na} studies and {b} is raised in {nb} papers."
                comp = self._comp(topic, methods=[a], aspects=[b])
            elif fa == "tasks":
                title, miss = f"{b.capitalize()} in {a} for {topic}", f"Evaluating {d} for {a} in {topic}."
                stmt = f"{a.title()} is studied in {na} primary papers on {topic}, yet none evaluates {b}; {b} is raised in {nb} papers."
                comp = self._comp(topic, tasks=[a], aspects=[b])
            else:
                title, miss = f"{b.title()} analysis on {a}", f"Analysing {d} on the {a} dataset."
                stmt = f"The {a} dataset is used in {na} studies on {topic}, but no study analyses {b} on it."
                comp = self._comp(topic, datasets=[a], aspects=[b])
            known, why = f"{a} is established in {topic} ({na} primary studies).", why[0].upper() + why[1:] + "."
        elif fa == "methods" and fb == "tasks":
            title, miss = f"{a.capitalize()} for {b}", f"Applying {a} to {b} in {topic} and comparing with the current standard."
            stmt = f"{a.title()} appears in {na} studies and {b} in {nb} studies on {topic}, but no retrieved study applies {a} to {b}."
            known, why, comp = f"{a} and {b} are each well studied in {topic}.", f"Transferring a proven method to a well-studied task is a low-risk, publishable contribution.", self._comp(topic, methods=[a], tasks=[b])
        else:
            title, miss = f"{a.capitalize()} on {b}", f"Evaluating {a} on the {b} dataset and comparing with published results."
            stmt = f"{a.title()} (used in {na} papers) has not been evaluated on {b} (used in {nb} papers) in the retrieved {topic} literature."
            known, why, comp = f"{a} and {b} are both established.", "A new method–data pairing shows whether published results transfer.", self._comp(topic, methods=[a], datasets=[b])
        items = [(p, f"uses {a}") for p in sorted(idx[fa][a]["study"])[:2]] + [(p, f"{'raises' if fb == 'aspects' else 'uses'} {b}") for p in sorted(idx[fb][b]["all"])[:2]]
        return {"type": "CONCEPT_GAP", "score": raw * W["CONCEPT_GAP"], "title": title, "statement": stmt, "known": known, "missing": miss, "why": why,
                "components": comp, "evidence": self._ev(kg, items), "rationale": f"{a} × {b}: co-occurrence in primary studies = {co}."}

    def _untested(self, kg, idx, topic, rnd):
        out = []
        for a, ia in idx["aspects"].items():
            nr, ns = len(ia["review"]), len(ia["study"])
            if nr >= 2 and ns <= 1 + rnd:
                d, why = ASP(a)[1], ASP(a)[2]
                out.append({"type": "UNTESTED_CONCERN", "score": clip(0.4 + 0.12 * nr - 0.1 * ns) * W["UNTESTED_CONCERN"],
                            "title": f"Empirical evaluation of {a} in {topic}",
                            "statement": f"{a.title()} is raised as an open concern in {nr} review/position papers on {topic}, but only {ns} primary stud{'y' if ns == 1 else 'ies'} examine{'s' if ns == 1 else ''} it empirically.",
                            "known": f"Reviews agree that {a} matters for {topic}.", "missing": f"Primary empirical studies that measure {d}.",
                            "why": why[0].upper() + why[1:] + ".", "components": self._comp(topic, aspects=[a]),
                            "evidence": self._ev(kg, [(p, "review raising concern") for p in sorted(ia["review"])[:3]]), "rationale": f"{nr} reviews vs {ns} primary studies."})
        return sorted(out, key=lambda c: -c["score"])[:3]

    def _contradictions(self, kg, topic):
        out = []
        for u, v, a in kg.edges_of("CONTRADICTS"):
            pa, pb = kg.papers[u[6:]], kg.papers[v[6:]]
            sa, sb = pa["structured"], pb["structured"]
            shared = [x for x in (a.get("shared") or "").split(", ") if x] or sa["methods"][:1]
            ctx = ", ".join(shared) or topic
            out.append({"type": "CONTRADICTION", "score": clip(a["weight"]) * W["CONTRADICTION"],
                        "title": f"Conflicting results on {ctx}", "statement": f"Two primary studies on {ctx} reach opposite conclusions: “{pa['title'][:90]}” vs “{pb['title'][:90]}”.",
                        "known": f"A: {sa['results'][:160]}  B: {sb['results'][:160]}", "missing": "Identify the data, setting or method differences that explain the disagreement (a replication or controlled comparison).",
                        "why": "Unresolved contradictions block reliable adoption.", "components": self._comp(topic, methods=shared[:1], datasets=(sb["datasets"][:1])),
                        "evidence": self._ev(kg, [(pa["pid"], "reports result A"), (pb["pid"], "reports opposing result B")]), "rationale": f"Opposite result polarity (score {a['weight']:.2f})."})
        return sorted(out, key=lambda c: -c["score"])[:3]

    def _limitations(self, kg, topic, rnd):
        items = []
        topic_terms = [t for t in re.findall(r"[a-z0-9]+", topic.lower()) if t not in {"in", "on", "for", "and", "with", "under", "using", "of", "to", "the", "a", "an"} and len(t) > 1]
        for pid, p in kg.papers.items():
            st = p.get("structured") or {}
            if st.get("paper_type") == "review":
                continue
            ptext = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
            if len(set(topic_terms)) >= 2:
                hits = sum(1 for term in set(topic_terms) if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", ptext))
                if hits < max(2, min(4, math.ceil(0.35 * len(set(topic_terms))))):
                    continue
            if "aai" in topic_terms and "animal assisted" not in topic and ("animal-assisted" in ptext or "animal assisted" in ptext):
                continue
            items += [(pid, t) for t in st.get("limitations", []) if len(tokens(t)) >= 6] + [(pid, t) for t in st.get("future_work", []) if len(tokens(t)) >= 6]
        items = items[:250]
        if len(items) < 2:
            return []
        S, thr = (lambda E: E @ E.T)(embeddings.embed([t for _, t in items])), 0.6 - 0.05 * rnd
        used, out = set(), []
        for i in range(len(items)):
            if i in used:
                continue
            grp = [j for j in range(len(items)) if S[i, j] >= thr and j not in used]
            pids = sorted({items[j][0] for j in grp})
            if len(pids) >= 2:
                used.update(grp)
                rep = items[i][1].rstrip(".")
                out.append({"type": "REPEATED_LIMITATION", "score": clip(len(pids) / 5) * W["REPEATED_LIMITATION"], "title": "Recurring limitation: " + " ".join(rep.split()[:8]) + "…",
                            "statement": f"{len(pids)} primary studies independently report the same limitation or open direction: “{rep[:220]}”.",
                            "known": "The limitation is acknowledged by several groups.", "missing": "A study that directly addresses this limitation and measures the improvement.",
                            "why": "Repeated acknowledgement indicates a shared, unsolved obstacle.", "components": self._comp(topic),
                            "evidence": self._ev(kg, [(items[j][0], "states limitation") for j in grp[:4]]), "rationale": "Concrete limitation repeated across papers."})
        return sorted(out, key=lambda c: -c["score"])[:2]

    def _polish(self, cands, topic):
        """With an LLM, rewrite title/statement/missing as precise, evidence-grounded text (no new facts allowed)."""
        if not (self.llm and self.llm.available and cands):
            return
        payload = "\n".join(f"[{i}] type={c['type']} | {c['statement']} | evidence: " + " || ".join(e["snippet"][:150] for e in c["evidence"][:3]) for i, c in enumerate(cands[:8]))
        r = self.llm.json(f"Topic: {topic}\nRewrite each candidate research gap so it is specific and researchable. Use ONLY facts in the candidate/evidence; add no new claims.\n{payload}\n"
                          'JSON: {"gaps":[{"title":"<=12 words","statement":"<=45 words: what is known, what is NOT studied","missing":"the concrete study that would fill it"}]} in the same order.')
        for c, g in zip(cands, (r or {}).get("gaps", [])):
            if isinstance(g, dict) and 15 < len(str(g.get("statement", ""))) < 450:
                c["raw_statement"] = c["statement"]
                c["title"], c["statement"], c["missing"] = str(g.get("title", c["title"]))[:140], g["statement"].strip(), str(g.get("missing", c["missing"]))[:300]
                c["id"] = "g" + hashlib.sha1((c["title"] + c["statement"]).encode()).hexdigest()[:8]
