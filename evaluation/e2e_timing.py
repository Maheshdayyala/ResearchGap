"""End-to-end latency measurement. Usage: python -m evaluation.e2e_timing "topic" """
import sys
import time

from agents import get_agents
from workflow import run_workflow

if __name__ == "__main__":
    t = time.time()
    s = run_workflow(get_agents(), 0, sys.argv[1], lambda st, p, m: print(f"[{time.time() - t:6.1f}s] {st}: {m}"))
    print(f"\nTotal runtime: {time.time() - t:.1f}s | papers={len(s['papers'])} | verified={bool(s.get('verified_gap'))}")
    print("Compare with the 1-hour manual literature review baseline measured with human participants.")
