"""Experiment Planning Agent: verified gap -> a grounded, runnable experiment plan.
Datasets come from a curated registry of real datasets (with links and access terms), datasets used in the retrieved literature,
and an optional live Hugging Face search. Baselines come from the literature plus task/aspect-standard methods. Metrics are chosen
by task and by the aspect the gap is about. Nothing is invented, and unsuitable items are left out instead of padded in."""
import logging
import re
from collections import Counter

import requests

from agents import lexicon as lx
from utils import uniq

log = logging.getLogger(__name__)
GENERIC_DATASET_TERMS = {"llm", "language model", "large language model", "machine learning", "deep learning", "model", "ai", "artificial intelligence"}
PROTOCOL = {
    "fairness": "Stratify the held-out set by sensitive attributes (e.g. sex, age group, race/ethnicity where available) and report per-group metrics and gaps with confidence intervals.",
    "privacy": "Run membership-inference and attribute-inference attacks against trained models; report attack success against the privacy budget and the utility cost.",
    "explainability": "Generate attributions with the method and with standard explainers; quantify faithfulness (deletion/insertion curves) and, if possible, collect expert ratings.",
    "robustness": "Evaluate under simulated and natural distribution shift (new site/time period, noise, perturbation) and report the drop relative to in-distribution performance.",
    "generalizability": "Train on one site/cohort and test on held-out external sites (leave-one-site-out); report internal vs external performance.",
    "data scarcity": "Train with 1%, 5%, 10%, 50%, 100% of labels; plot learning curves for the method and each baseline.",
    "uncertainty": "Report calibration (ECE, reliability diagrams) and selective-prediction curves before and after calibration.",
    "real-world deployment": "Run a retrospective-to-prospective simulation (temporal split) and measure latency and workflow cost; flag what a true prospective study would add.",
    "efficiency": "Benchmark latency, memory and energy on the target hardware at matched accuracy.",
}


class ExperimentPlanningAgent:
    def __init__(self, llm=None):
        self.llm = llm

    # ---------------------------------------------------------------- public
    def plan(self, gap, papers, novelty=None):
        cand = self._candidates(gap, papers)
        plan = self._heuristic(gap, cand)
        if self.llm and self.llm.available:
            plan = self._llm_refine(gap, cand, plan)
        plan["novelty_verdict"] = (novelty or {}).get("verdict")
        return plan

    # ---------------------------------------------------------------- candidates
    @staticmethod
    def _context(gap):
        c = gap.get("components", {})
        return " ".join([gap.get("title", ""), gap.get("statement", ""), gap.get("missing", ""), c.get("topic", "")] +
                        [x for k in ("methods", "aspects", "tasks", "datasets") for x in c.get(k, [])]).lower()

    def _candidates(self, gap, papers):
        c, ctx = gap.get("components", {}), self._context(gap)
        studies = [p for p in papers if (p.get("structured") or {}).get("paper_type") != "review"] or papers
        st = lambda p: p.get("structured") or {}
        # ---- modality & tasks
        mod_scores = {m: len(re.findall(rx, ctx, re.I)) for m, rx in lx.MODALITY_RE.items()}
        if not any(mod_scores.values()):
            blob = " ".join(p["title"] + " " + p["abstract"][:300] for p in studies[:15]).lower()
            mod_scores = {m: len(re.findall(rx, blob, re.I)) for m, rx in lx.MODALITY_RE.items()}
        modality = max(mod_scores, key=mod_scores.get) if any(mod_scores.values()) else None
        tasks = c.get("tasks") or lx.tasks_in(ctx) or [t for t, _ in Counter(t for p in studies for t in st(p).get("tasks", [])).most_common(1)]
        aspects = c.get("aspects") or []
        scoped_tasks = {t for t in tasks if t in lx.TASK_BASELINES}
        if scoped_tasks:
            # For specialized tasks (such as citation verification), source datasets and
            # baselines only from papers that actually study that task.
            studies = [p for p in studies if set(st(p).get("tasks", [])) & scoped_tasks]

        # ---- datasets
        corpus_ds = Counter(d for p in studies for d in st(p).get("datasets", []))
        scored = {}
        for d in lx.DATASETS:
            hits = [k for k in d["kw"] if k.lower() not in GENERIC_DATASET_TERMS and k.lower() in ctx]
            if scoped_tasks and d["modality"] != modality and not any(x.lower() == d["name"].lower() for x in c.get("datasets", [])):
                continue
            n_corpus = max([corpus_ds[a] for a, dd in lx.DATASET_ALIASES.items() if dd is d and a in corpus_ds] or [0])
            in_gap = any(lx.DATASET_ALIASES.get(x.lower()) is d for x in c.get("datasets", []))
            score = len(hits) + 2 * min(n_corpus, 3) + (2 if modality and d["modality"] == modality else 0) * bool(hits or n_corpus) + 4 * in_gap
            # A single broad keyword such as "LLM" is not enough to recommend an unrelated benchmark.
            if len(hits) >= 2 or n_corpus or in_gap:
                why = ([f"used in {n_corpus} retrieved paper(s)"] if n_corpus else []) + ([f"matches: {', '.join(hits[:3])}"] if hits else []) + (["named in the gap"] if in_gap else [])
                scored[d["name"]] = (score, {"name": d["name"], "url": d["url"], "access": d["access"], "reason": "; ".join(why) + f". {d['desc']}.", "from_corpus": bool(n_corpus), "source": "registry"})
        datasets = [v for _, v in sorted(scored.values(), key=lambda x: -x[0])][:4]
        known = {d["name"] for d in datasets}
        for name, n in corpus_ds.most_common(6):          # corpus datasets not in the registry
            if name not in lx.DATASET_ALIASES and len(datasets) < 5 and n >= 2:
                datasets.append({"name": name, "url": "", "access": "unknown, verify", "reason": f"used in {n} retrieved papers but not in the curated registry; verify availability", "from_corpus": True, "source": "literature"})
        # Do not pad the plan with loosely related datasets. An empty/short list is safer than a false fit.

        # ---- baselines
        base, used = [], {m.lower() for m in c.get("methods", [])}
        meth = Counter(m for p in studies for m in st(p).get("methods", []) if m not in lx.GENERIC_BASELINES and m not in used)
        for m, n in meth.most_common(3):
            ex = next(p["title"] for p in studies if m in st(p).get("methods", []))
            base.append({"name": m, "reason": f"used in {n} retrieved paper(s), e.g. “{ex[:70]}”", "source": "literature"})
        for m_ in c.get("methods", []):
            base += [{"name": b, "reason": f"reference comparison for {m_}", "source": "standard"} for b in lx.METHOD_BASELINES.get(m_, [])[:3]]
        for a in aspects:
            base += [{"name": b, "reason": f"standard comparison for {a}", "source": "standard"} for b in lx.ASPECT_BASELINES.get(a, [])[:3]]
        for t in tasks:
            base += [{"name": b, "reason": f"comparison baseline for {t}; verify fit against the evidence papers", "source": "standard"}
                     for b in lx.TASK_BASELINES.get(t, [])[:3]]
        base += [{"name": b, "reason": f"standard {modality} baseline", "source": "standard"} for b in lx.MODALITY_BASELINES.get(modality, [])[:3]] if modality else []
        baselines = list({b["name"].lower(): b for b in base}.values())[:9]

        # ---- metrics
        metrics = []
        for t in tasks[:2]:
            metrics += [{"name": m, "reason": f"primary metric for {t}"} for m in lx.TASK_METRICS.get(t, [])[:3]]
        for a in aspects:
            metrics += [{"name": m, "reason": f"measures {a}"} for m in lx.ASPECT_METRICS.get(a, [])[:3]]
        # If the task is not identifiable, leave metrics empty rather than inventing a generic evaluation protocol.
        metrics = list({m["name"]: m for m in metrics}.values())[:9]
        return {"modality": modality, "tasks": tasks, "aspects": aspects, "datasets": datasets, "baselines": baselines, "metrics": metrics}

    @staticmethod
    def _hf_search(gap):
        c = gap.get("components", {})
        q = " ".join([c.get("topic", "")] + [lx.ASPECTS[a][3].split()[0] for a in c.get("aspects", []) if a in lx.ASPECTS])[:80]
        try:
            r = requests.get("https://huggingface.co/api/datasets", params={"search": q, "limit": 3, "sort": "downloads"}, timeout=8)
            r.raise_for_status()
            return [{"name": d["id"], "url": f"https://huggingface.co/datasets/{d['id']}", "access": "open (check licence on the page)",
                     "reason": "found via Hugging Face search for the topic; verify it fits the task before use", "from_corpus": False, "source": "huggingface"} for d in r.json()]
        except Exception as e:
            log.info("HF dataset search unavailable: %s", e)
            return []

    # ---------------------------------------------------------------- plan assembly
    def _heuristic(self, gap, cand):
        c, topic = gap.get("components", {}), gap.get("components", {}).get("topic", "the target domain")
        m = (c.get("methods") or [""])[0]
        asp = cand["aspects"][0] if cand["aspects"] else ""
        task = cand["tasks"][0] if cand["tasks"] else "the target task"
        focus = m or (c.get("datasets") or [""])[0] or task
        where = topic if task in topic.lower() or task == "the target task" else f"{task} in {topic}"
        if m and asp:
            rq = f"When {m} is applied to {where}, how does it fare on {asp} ({lx.ASPECTS[asp][1]}) compared with standard baselines?"
        elif asp:
            rq = f"How well do current approaches to {where} handle {asp} ({lx.ASPECTS[asp][1]})?"
        else:
            rq = gap.get("missing") or gap["statement"]
        pm = (cand["metrics"][0]["name"] if cand["metrics"] else "the primary metric")
        am = next((x["name"] for x in cand["metrics"] if asp and asp in x["reason"]), pm)
        hyp = f"H1: {focus} produces a measurable difference in {am} relative to the strongest baseline. H0: no difference (paired bootstrap, 95% CI, α = 0.05)."
        ds, bl, mt = cand["datasets"], cand["baselines"], cand["metrics"]
        steps = [
            (f"Data: obtain {', '.join(d['name'] for d in ds[:2])} ({'; '.join(uniq(d['access'] for d in ds[:2]))}); " if ds else "Data: no suitable dataset was identified automatically; select a task-specific benchmark and verify its licence and labels before proceeding; ")
            + "define train/validation/test splits at the independent unit (for example, keep claims and their source papers in one split) to prevent information leakage.",
            f"Baselines: reproduce {', '.join(b['name'] for b in bl[:4]) or 'the strongest published baselines'} with their published settings.",
            f"Method: implement/apply {focus} for {task}; tune hyper-parameters on validation data only.",
            (f"Evaluation: report {', '.join(x['name'] for x in mt[:5])}." if mt else "Evaluation: select task-appropriate metrics after the exact prediction/generation task and label format are defined."),
        ]
        if asp:
            steps.append(f"{asp.title()} protocol: {PROTOCOL.get(asp, 'Design an explicit test of ' + asp + '.')}")
        steps += ["Statistics: ≥5 random seeds, mean ± std, paired bootstrap or permutation test with multiple-comparison correction.",
                  "Analysis: ablate each component of the method and examine failure cases against the evidence papers' reported limitations."]
        risks = [f"{d['name']}: {d['access']}" for d in ds if not d["access"].startswith("open")][:2] + ["Compute budget and baseline reproducibility", "Results may not transfer beyond the chosen datasets; state this scope explicitly"]
        if asp == "fairness":
            risks.append("Sensitive attributes may be missing or noisy in the chosen dataset")
        return {"research_question": rq, "hypothesis": hyp, "methodology": steps, "datasets": ds, "baselines": bl, "metrics": mt,
                "ablations": ["Remove each novel component", "Vary training-set size", "Evaluate on a second dataset/site"], "risks": risks,
                "modality": cand["modality"], "generated_by": "rules"}

    def _llm_refine(self, gap, cand, plan):
        r = self.llm.json(
            f"Research gap: {gap['statement']}\nMissing study: {gap.get('missing', '')}\n"
            f"CANDIDATE DATASETS: {[d['name'] for d in cand['datasets']]}\nCANDIDATE BASELINES: {[b['name'] for b in cand['baselines']]}\n"
            f"CANDIDATE METRICS: {[m['name'] for m in cand['metrics']]}\n"
            "Write an experiment plan. Select datasets/baselines/metrics ONLY from the candidate lists (never invent names). "
            'JSON: {"research_question":"","hypothesis":"","dataset_names":[],"baseline_names":[],"metric_names":[],"methodology":["concrete step naming the chosen items"],"ablations":[],"risks":[]}', max_tokens=1800)
        if not r or not r.get("methodology"):
            return plan
        pick = lambda items, names: [i for i in items if i["name"] in set(names)] or items
        plan.update(research_question=r.get("research_question") or plan["research_question"], hypothesis=r.get("hypothesis") or plan["hypothesis"],
                    methodology=[str(s) for s in r["methodology"]][:9], datasets=pick(plan["datasets"], r.get("dataset_names", [])),
                    baselines=pick(plan["baselines"], r.get("baseline_names", [])), metrics=pick(plan["metrics"], r.get("metric_names", [])),
                    ablations=r.get("ablations") or plan["ablations"], risks=r.get("risks") or plan["risks"], generated_by="llm (grounded in candidate lists)")
        return plan
