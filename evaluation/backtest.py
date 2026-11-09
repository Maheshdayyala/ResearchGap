"""Historical time-split back-testing of the Novelty Verification Agent (Table 4).
The corpus is restricted to papers published before CUTOFF; known breakthrough papers from 2024-2025 are then scored as
"gap statements". A breakthrough labelled novel on the pre-cutoff corpus is a true positive.
Usage: python -m evaluation.backtest topic.json   with {"topic": "...", "breakthroughs": ["title or abstract", ...]}"""
import json
import sys

from agents import get_agents
from settings import settings

CUTOFF = 2023


def run(path):
    cfg, ag = json.load(open(path)), get_agents()
    corpus = [p for p in ag.search.fetch(cfg["topic"]) if (p.get("year") or 0) < CUTOFF and p.get("abstract")]
    for p, st in zip(corpus, ag.understand.extract_many(corpus)):
        p["structured"] = st
    tp = 0
    for text in cfg["breakthroughs"]:
        gap = {"id": "bt", "type": "CUSTOM", "statement": text, "components": {"methods": [], "datasets": [], "topic": cfg["topic"]}, "evidence": []}
        res = ag.novelty.verify(gap, corpus, expand=False)   # expand=False keeps the corpus strictly pre-cutoff
        tp += res["score"] >= settings.NOVELTY_THRESHOLD
        print(f"{res['score']:.2f}  {'NOVEL' if res['score'] >= settings.NOVELTY_THRESHOLD else 'not novel'}  {text[:80]}")
    n = len(cfg["breakthroughs"])
    print(f"\nTrue-positive rate on {n} breakthrough cases: {tp / n:.2f}  (indicative only: very small sample)")


if __name__ == "__main__":
    run(sys.argv[1])
