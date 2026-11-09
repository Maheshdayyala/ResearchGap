"""Paper Understanding Agent: abstract -> structured JSON (goal, methods, datasets, tasks, evaluated aspects, results, limitations)."""
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from agents import lexicon as lx
from utils import sentences, tokens, uniq

log = logging.getLogger(__name__)

SYSTEM = ("You are a scientific information-extraction engine. Extract ONLY what is explicitly stated; never invent datasets, "
          "methods or numbers. Use lowercase canonical names. Do not list generic terms like 'machine learning' or 'ai' as methods.")
PROMPT = """Title: {title}
Abstract: {abstract}

Return JSON with keys:
"goal": one sentence research goal,
"methods": specific named methods/models/algorithms (max 8),
"datasets": named datasets/benchmarks/cohorts actually used (max 8),
"results": one or two sentences of key findings,
"limitations": concrete limitations stated or clearly implied, e.g. single-site data, small sample, no external validation (max 4),
"future_work": concrete future directions (max 3)."""

# a limitation must be concrete: contain one of these cues and NOT be a generic hedge
LIMIT_CUE = re.compile(r"\b(limited to|limitation|lack(s|ing)? of|not (been )?(evaluated|validated|tested|explored|investigated|addressed)|only (on|in|for|one|a single)|single[- ](site|center|centre|institution|dataset)|"
                       r"small (sample|dataset|cohort)|little (research|work|evidence)|few (studies|works)|remains? (unclear|unexplored|unresolved|open)|"
                       r"unexplored|under-?explored|no (study|studies|work)|has not|have not|fail(s|ed)? to|cannot|does not (generali[sz]e|account))\b", re.I)
HEDGE = re.compile(r"\b(there are also possibilities|may (continue|exacerbate)|potential applications|transformative|revolution|promising|offers? (great|new) )", re.I)
FUTURE_CUE = re.compile(r"\b(future (work|research|studies|directions?)|in the future|we plan to|further (research|work|investigation|studies)|remains? to be)\b", re.I)
RESULT_CUE = re.compile(r"\b(outperform|achiev|improv|result(s)? (show|indicate|demonstrate)|show(s|ed)? that|demonstrat|accuracy|auc|f1|reduc|significant|found that)", re.I)
GOAL_CUE = re.compile(r"\b(we propose|we present|we introduce|this (paper|study|work|article)|aims? to|we (study|investigate|develop|explore|evaluate))\b", re.I)
PROPER_DS = re.compile(r"((?:[A-Z][\w\-.]*\s){0,3}[A-Z][\w\-.]*)\s+(?:dataset|data set|benchmark|corpus|cohort|database)s?\b")


def _hits(text, vocab):
    low = text.lower()
    return [v for v in vocab if re.search(r"(?<![\w-])" + re.escape(v) + r"(?![\w-])", low)]


class PaperUnderstandingAgent:
    def __init__(self, llm=None):
        self.llm = llm

    def extract(self, paper):
        text, title = (paper.get("abstract") or "").strip(), paper.get("title", "")
        out = None
        if self.llm and self.llm.available and len(text) > 80:
            out = self.llm.json(PROMPT.format(title=title, abstract=text[:3500]), SYSTEM)
        res = self._normalise(out, title, text) if out else self._heuristic(title, text)
        # rule-based facets are always computed so every paper is comparable regardless of LLM availability
        blob = f"{title}. {text}"
        res["aspects"] = lx.aspects_in(blob)
        res["tasks"] = lx.tasks_in(blob)
        res["paper_type"] = "review" if lx.is_review(title, text) else "study"
        res["extraction"] = "llm" if out else "heuristic"
        return res

    def extract_many(self, papers, workers=6):
        with ThreadPoolExecutor(max_workers=workers if self.llm and self.llm.available else 1) as ex:
            return list(ex.map(self.extract, papers))

    # ---------------------------------------------------------------
    @staticmethod
    def _clean_methods(items):
        out = [lx.canon_method(m) for m in items if isinstance(m, str)]
        return uniq([m for m in out if m and m not in lx.GENERIC_METHODS and not m.endswith("-based") and len(m) > 2])

    @staticmethod
    def _clean_datasets(items):
        out = []
        for d in items:
            n = re.sub(r"\s+", " ", str(d).strip().lower()).strip(" .,;:()")
            n = re.sub(r"^(the|a|an)\s+", "", n)
            if n in lx.DATASET_ALIASES or (n and n.split()[0] not in lx.BAD_DATASET_WORDS and re.search(r"\d|-", n) and len(n) > 2):
                out.append(n)
        return uniq(out)

    def _normalise(self, o, title, text):
        sl = lambda k, n: [str(x).strip() for x in (o.get(k) or []) if str(x).strip()][:n]
        h = self._heuristic(title, text)
        lim = [x for x in sl("limitations", 4) if not HEDGE.search(x)]
        return {"goal": str(o.get("goal", "")).strip() or h["goal"], "methods": self._clean_methods(o.get("methods") or [])[:8] or h["methods"],
                "datasets": self._clean_datasets(o.get("datasets") or [])[:8], "results": str(o.get("results", "")).strip(),
                "limitations": lim, "future_work": sl("future_work", 3), "topics": h["topics"]}

    def _heuristic(self, title, text):
        sents = sentences(text)
        pick = lambda rx, n: [s for s in sents if rx.search(s) and not HEDGE.search(s)][:n]
        goal = (pick(GOAL_CUE, 1) or sents[:1] or [title])[0]
        methods = self._clean_methods(_hits(f"{title}. {text}", lx.METHOD_VOCAB) + [a for a in lx.ALIASES if re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", f"{title} {text}".lower()) and len(a) > 2])
        ds = [d for d in lx.DATASET_ALIASES if re.search(r"(?<![\w-])" + re.escape(d) + r"(?![\w-])", text.lower())]
        ds += [m.group(1) for m in PROPER_DS.finditer(text)]
        topics = [w for w, _ in Counter(tokens(title) + tokens(text)).most_common(5)]
        return {"goal": goal, "methods": methods[:8], "datasets": self._clean_datasets(ds)[:8], "results": " ".join(pick(RESULT_CUE, 2)),
                "limitations": [s for s in sents if LIMIT_CUE.search(s) and not HEDGE.search(s) and len(tokens(s)) >= 4][:3],
                "future_work": pick(FUTURE_CUE, 2), "topics": topics}
