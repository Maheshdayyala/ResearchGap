"""LLM-as-a-judge rubrics (extraction audit, gap specificity/faithfulness, experiment-plan alignment).
Requires an LLM key. Use a small model for volume (LLM_MODEL=claude-haiku-5-5 / gpt-4o-mini).
Ragas / DeepEval can be layered on top of the same inputs; this module has no hard dependency on them."""
import json
import statistics

from agents import get_agents

GAP_RUBRIC = """Score the research-gap statement 1-5 on each: specificity (not vague/generic), faithfulness (supported by the evidence
snippets, no invented claims), researchability (a concrete study could address it). JSON: {"specificity":n,"faithfulness":n,"researchability":n}"""
PLAN_RUBRIC = """Score 1-5: (1) datasets are existing and publicly accessible, (2) baselines come from the literature corpus,
(3) metrics are standard for the task. JSON: {"datasets":n,"baselines":n,"metrics":n}"""


def judge_gaps(candidates):
    llm, rows = get_agents().llm, []
    for c in candidates:
        r = llm.json(f"{GAP_RUBRIC}\nGap: {c['statement']}\nEvidence: " + " | ".join(e["snippet"][:200] for e in c["evidence"]))
        if r:
            rows.append(r)
    return {k: round(statistics.mean(x[k] for x in rows), 2) for k in ("specificity", "faithfulness", "researchability")} if rows else {}


def judge_plan(plan, corpus_methods):
    r = get_agents().llm.json(f"{PLAN_RUBRIC}\nCorpus methods: {corpus_methods}\nPlan: {json.dumps(plan)[:3000]}")
    return r or {}


def extraction_audit(papers):
    """Build 'gold' JSON with the judge model and compare field-level overlap against the agent's extraction."""
    ag, scores = get_agents(), []
    for p in papers:
        gold = ag.llm.json(f"Title: {p['title']}\nAbstract: {p['abstract'][:3000]}\nExtract JSON with keys methods, datasets (lists of lowercase names).")
        pred = p.get("structured") or {}
        if not gold:
            continue
        for k in ("methods", "datasets"):
            g, q = set(map(str.lower, gold.get(k, []))), set(pred.get(k, []))
            tp = len(g & q)
            prec, rec = tp / len(q) if q else 0, tp / len(g) if g else 0
            scores.append({"field": k, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0})
    return {f: round(statistics.mean(s["f1"] for s in scores if s["field"] == f), 3) for f in ("methods", "datasets")} if scores else {}
