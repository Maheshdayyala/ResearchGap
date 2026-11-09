"""LangGraph orchestration: stateful routing + retry loops across the agents (see Figure 2 of the design document)."""
import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from settings import settings

log = logging.getLogger(__name__)


class State(TypedDict, total=False):
    search_id: int
    query: str
    current_query: str
    retries: int
    papers: List[Dict[str, Any]]
    source_status: Dict[str, str]
    kg: Any
    candidates: List[Dict[str, Any]]
    partials: List[Dict[str, Any]]       # gaps judged PARTIALLY_EXPLORED
    unverified: List[Dict[str, Any]]     # insufficient evidence; eligible only for a provisional plan
    plan_gap: Optional[Dict[str, Any]]
    corpus_stats: Dict[str, Any]
    evaluated: List[Dict[str, Any]]      # every gap that went through novelty verification (with its result)
    detect_round: int
    verified_gap: Optional[Dict[str, Any]]
    novelty: Optional[Dict[str, Any]]
    plan: Optional[Dict[str, Any]]
    report_md: str
    status: str


def build_workflow(ag, emit: Callable[[str, int, str], None] = lambda *a: None):
    def step(stage, pct, msg):
        emit(stage, pct, msg)

    def search(s):
        q = s.get("current_query") or s["query"]
        step("search", 10 + 3 * s.get("retries", 0), f"Searching literature: “{q}”")
        new, status = ag.search.fetch_ex(q)
        merged = ag.search.deduplicate(list(s.get("papers", [])) + new)
        merged = [p for p in ag.search.rank(s["query"], merged) if p.get("abstract")][: settings.MAX_PAPERS]
        return {"papers": merged, "source_status": status}

    def refine(s):
        r = s.get("retries", 0)
        nq = ag.search.refine_query(s["query"], r)
        step("search", 12 + 3 * r, f"Too few papers ({len(s['papers'])}); refining query → “{nq}”")
        return {"current_query": nq, "retries": r + 1}

    def route_search(s):
        if len(s["papers"]) < settings.MIN_PAPERS and s.get("retries", 0) < settings.MAX_REFINEMENTS:
            return "refine"
        return "understand" if s["papers"] else "report"

    def understand(s):
        step("understand", 30, f"Extracting structure from {len(s['papers'])} papers")
        for p, st in zip(s["papers"], ag.understand.extract_many(s["papers"])):
            p["structured"] = st
        n, rev = len(s["papers"]), sum(p["structured"]["paper_type"] == "review" for p in s["papers"])
        warn = []
        if n and rev / n > 0.5:
            warn.append(f"{rev} of {n} retrieved papers are reviews/surveys, so the topic is probably too broad. Narrow it (e.g. add a task, method or population) for sharper gaps.")
        return {"papers": s["papers"], "corpus_stats": {"papers": n, "reviews": rev, "studies": n - rev, "warnings": warn}}

    def graph(s):
        step("graph", 45, "Building knowledge graph")
        return {"kg": ag.kg_builder.build(s["search_id"], s["papers"], query=s["query"])}

    def detect(s):
        rnd = s.get("detect_round", 0)
        step("gaps", 55 + 3 * rnd, f"Detecting research gaps (round {rnd + 1})")
        excl = {e["statement"] for e in s.get("evaluated", [])}
        return {"candidates": ag.gap.detect(s["kg"], rnd, excl), "detect_round": rnd + 1}

    def route_detect(s):
        if s["candidates"]:
            return "verify"
        return "detect" if s["detect_round"] < settings.MAX_DETECT_ROUNDS else "report"

    def verify(s):
        gap, rest = s["candidates"][0], s["candidates"][1:]
        step("novelty", 68, f"Prior-art check {len(s.get('evaluated', [])) + 1}: {gap['title'][:70]}")
        res = ag.novelty.verify(gap, s["papers"])
        ev = list(s.get("evaluated", [])) + [{**gap, "novelty": res}]
        out = {"candidates": rest, "evaluated": ev}
        if res["accepted"]:
            out.update(verified_gap=gap, novelty=res)
        elif res["verdict"] == "PARTIALLY_EXPLORED":
            out["partials"] = list(s.get("partials", [])) + [{"gap": gap, "novelty": res}]
        elif res["verdict"] == "INSUFFICIENT_EVIDENCE":
            out["unverified"] = list(s.get("unverified", [])) + [{"gap": gap, "novelty": res}]
        return out

    def route_verify(s):
        if s.get("verified_gap"):
            return "plan"
        out_of_budget = len(s.get("evaluated", [])) >= settings.MAX_VERIFY
        if s["candidates"] and not out_of_budget:
            return "verify"
        if s["detect_round"] < settings.MAX_DETECT_ROUNDS and not out_of_budget:
            return "detect"
        return "fallback" if (s.get("partials") or s.get("unverified")) else "report"

    def fallback(s):
        pool = s.get("partials") or s.get("unverified") or []
        best = max(pool, key=lambda x: x["novelty"].get("score", 0))
        if best["novelty"]["verdict"] == "PARTIALLY_EXPLORED":
            return {"verified_gap": best["gap"], "plan_gap": best["gap"], "novelty": best["novelty"]}
        return {"plan_gap": best["gap"], "novelty": best["novelty"]}

    def plan(s):
        step("plan", 85, "Planning experiments")
        gap = s.get("plan_gap") or s.get("verified_gap")
        plan = ag.planner.plan(gap, s["papers"], s["novelty"])
        plan["provisional"] = not bool(s.get("verified_gap"))
        return {"plan": plan}

    def report(s):
        step("report", 95, "Compiling report")
        kg = s.get("kg")
        ctx = {"query": s["query"], "papers": s.get("papers", []), "topics": kg.topics if kg else [],
               "candidates": sorted(s.get("evaluated", []), key=lambda c: (not c["novelty"]["accepted"], -c["novelty"]["score"])), "verified_gap": s.get("verified_gap"),
               "novelty": s.get("novelty"), "plan": s.get("plan"), "corpus_stats": s.get("corpus_stats")}
        return {"report_md": ag.report.generate(ctx), "status": "done" if s.get("papers") else "failed"}

    g = StateGraph(State)
    for name, fn in [("search", search), ("refine", refine), ("understand", understand), ("graph", graph), ("detect", detect),
                     ("verify", verify), ("fallback", fallback), ("plan", plan), ("report", report)]:
        g.add_node(name, fn)
    g.add_edge(START, "search")
    g.add_conditional_edges("search", route_search, {"refine": "refine", "understand": "understand", "report": "report"})
    g.add_edge("refine", "search")
    g.add_edge("understand", "graph")
    g.add_edge("graph", "detect")
    g.add_conditional_edges("detect", route_detect, {"verify": "verify", "detect": "detect", "report": "report"})
    g.add_conditional_edges("verify", route_verify, {"plan": "plan", "verify": "verify", "detect": "detect", "fallback": "fallback", "report": "report"})
    g.add_edge("fallback", "plan")
    g.add_edge("plan", "report")
    g.add_edge("report", END)
    return g.compile()


def run_workflow(ag, search_id, query, emit=lambda *a: None):
    wf = build_workflow(ag, emit)
    return wf.invoke({"search_id": search_id, "query": query, "current_query": query, "retries": 0, "papers": [],
                       "evaluated": [], "partials": [], "unverified": [], "detect_round": 0, "verified_gap": None, "plan_gap": None, "novelty": None, "plan": None},
                     config={"recursion_limit": 120})
