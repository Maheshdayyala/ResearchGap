"""Report Generation Module: merges agent outputs into an evidence-supported Markdown/HTML report."""
from collections import defaultdict
from datetime import datetime

import bleach
import markdown as md

ALLOWED = ["h1", "h2", "h3", "h4", "p", "ul", "ol", "li", "strong", "em", "code", "pre", "table", "thead", "tbody", "tr", "th", "td", "a", "blockquote", "hr", "br"]
LABEL = {"NOVEL": "Potentially unexplored", "PARTIALLY_EXPLORED": "Partially explored", "ALREADY_ADDRESSED": "Already addressed", "INSUFFICIENT_EVIDENCE": "Insufficient evidence"}


def _cite(p):
    a = (p.get("authors") or ["Unknown"])
    link = f" https://doi.org/{p['doi']}" if p.get("doi") else (f" {p['url']}" if p.get("url") else "")
    return f"{a[0]}{' et al.' if len(a) > 1 else ''} ({p.get('year') or 'n.d.'}). {p['title']}. {p.get('venue') or ''}.{link}".replace(" .", ".")


class ReportGenerator:
    def generate(self, ctx):
        q, papers = ctx["query"], ctx.get("papers", [])
        gap, nov, plan, cs = ctx.get("verified_gap"), ctx.get("novelty"), ctx.get("plan"), ctx.get("corpus_stats") or {}
        L = [f"# Research Gap Report: {q}", f"*Generated {datetime.now():%Y-%m-%d %H:%M} by ResearchGapAI · {len(papers)} papers analysed" + (f" ({cs.get('studies')} primary studies, {cs.get('reviews')} reviews)" if cs else "") + "*", ""]
        L += [f"> **Note:** {w}" for w in cs.get("warnings", [])]
        L += ["## 1. Recommended research gap"]
        if gap:
            L += [f"### {gap['title']}", gap["statement"], "", f"- **Already known:** {gap.get('known', '')}", f"- **Not yet studied:** {gap.get('missing', '')}", f"- **Why it matters:** {gap.get('why', '')}", ""]
            L += [f"**Novelty verdict: {LABEL.get(nov['verdict'], nov['verdict'])}** (confidence {nov['confidence']}). {nov['explanation']}", ""]
        else:
            if plan and plan.get("provisional"):
                L += ["No candidate was verified as novel in this run. A provisional exploration plan is provided for a direction with insufficient prior-art evidence; it must not be described as a verified research gap.", ""]
            else:
                L += ["No candidate gap survived the prior-art check in this run. The candidates below show what was found and why each was rejected.", ""]
        L += ["## 2. Prior-art check"]
        if nov:
            L += [f"Concepts that had to co-occur: **{' + '.join(nov['core_concepts']) or 'n/a'}**. Papers compared: {nov['pool_size']}. {nov['confidence_reason']}", "", "Queries run:"]
            L += [f"- `{s['query']}` → {s.get('results', s.get('error'))} results" for s in nov["search_log"]]
            L += ["", "Closest existing work:"] + [f"- {c['title']} ({c.get('year') or 'n.d.'}, {c['type']}), matched: {', '.join(c['matched']) or 'none'}" for c in nov["closest"]] + [""]
        L += ["## 3. All candidate gaps", "| # | Gap | Type | Verdict |", "|---|---|---|---|"]
        for i, c in enumerate(ctx.get("candidates") or [], 1):
            n = c.get("novelty") or {}
            L += [f"| {i} | {c['title'][:90].replace('|', '/')} | {c['type'].replace('_', ' ').lower()} | {LABEL.get(n.get('verdict'), '–')} |"]
        L += [""]
        if plan:
            L += [("## 4. Provisional experiment plan" if plan.get("provisional") else "## 4. Experiment plan"),
                  ("**Status:** Exploratory only — novelty was not verified because prior-art evidence was insufficient." if plan.get("provisional") else ""),
                  f"**Research question:** {plan['research_question']}", "", f"**Hypothesis:** {plan['hypothesis']}", "", "**Procedure**"]
            L += [f"{i}. {s}" for i, s in enumerate(plan["methodology"], 1)]
            L += ["", "**Datasets**"] + [f"- [{d['name']}]({d['url']}) ({d['access']}): {d['reason']}" if d.get("url") else f"- {d['name']} ({d['access']}): {d['reason']}" for d in plan["datasets"]] or ["- none identified"]
            L += ["", "**Baselines**"] + [f"- {b['name']}: {b['reason']}" for b in plan["baselines"]]
            L += ["", "**Metrics**"] + [f"- {m['name']}: {m['reason']}" for m in plan["metrics"]]
            L += ["", "**Ablations:** " + "; ".join(plan["ablations"]), "", "**Risks:** " + "; ".join(plan["risks"]), ""]
        L += ["## 5. Literature map"]
        by, groups = {p["pid"]: p for p in papers}, defaultdict(list)
        for t in ctx.get("topics") or []:
            groups[t["label"]] += t["pids"]
        for label, pids in (groups.items() if groups else [("All papers", [p["pid"] for p in papers])]):
            L += [f"### {label} ({len(pids)})"] + [self._line(by[x]) for x in pids[:10]] + [""]
        L += ["## 6. Agent execution summary",
              "The following records describe what each agent actually processed in this analysis. They are execution traces and output counts, not benchmark-quality scores.",
              ""]
        for a in ctx.get("agent_evaluation") or []:
            L.append(f"- **{a.get('agent', 'Agent')}:** {a.get('status', 'unknown')} — {a.get('output', 'No output recorded')}. {a.get('note', '')}".strip())
        if not ctx.get("agent_evaluation"):
            L.append("- Per-agent execution details were not available for this run.")
        L += ["", "## 7. Scope and limitations",
              "- Search results depend on the coverage, indexing, rate limits and availability of the connected scholarly sources; retrieved papers may not represent the full literature.",
              "- Abstract-based extraction can miss details available only in full text. Acronyms and broad queries can retrieve papers from unrelated domains; define ambiguous terms in the query and inspect the paper list.",
              "- Dataset, baseline and metric suggestions must be checked for task fit and access before use. Generated plans may contain mismatches and should be reviewed by a domain expert.",
              "- A prior-art search can support or weaken a novelty claim but cannot prove that no publication anywhere has addressed a topic.", ""]
        L += ["## References"] + [f"{i}. {_cite(p)}" for i, p in enumerate(papers[:40], 1)]
        L += ["", "---", "*Gaps and verdicts are recommendations that need human validation. Absence in the searched databases is not proof of novelty.*"]
        return "\n".join(L)

    @staticmethod
    def _line(p):
        st = p.get("structured") or {}
        bits = [f"- **{p['title']}** ({p.get('year') or 'n.d.'}, {st.get('paper_type', 'study')})"]
        if st.get("methods"):
            bits.append(f"methods: {', '.join(st['methods'][:4])};")
        if st.get("datasets"):
            bits.append(f"datasets: {', '.join(st['datasets'][:3])};")
        if st.get("aspects"):
            bits.append(f"aspects: {', '.join(st['aspects'][:3])}")
        return " ".join(bits)

    @staticmethod
    def to_html(text):
        raw = md.markdown(text, extensions=["tables", "sane_lists"])
        return bleach.clean(raw, tags=ALLOWED, attributes={"a": ["href"]}, protocols=["http", "https"], strip=True)
