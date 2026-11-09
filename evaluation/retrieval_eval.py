"""Silver-standard retrieval evaluation (Table 4, Literature Search Agent).
Use a published survey's title as the query; its OpenAlex references are the silver ground truth.
Usage: python -m evaluation.retrieval_eval "Survey title 1" "Survey title 2" ..."""
import math
import sys

import requests

from agents import get_agents
from settings import settings
from utils import norm_doi


def survey_refs(title):
    r = requests.get("https://api.openalex.org/works", params={"search": title, "per-page": 1, "mailto": settings.CONTACT_EMAIL}, timeout=30).json()["results"][0]
    refs = [x.rsplit("/", 1)[-1] for x in r.get("referenced_works", [])]
    dois = {}
    for i in range(0, len(refs), 50):
        j = requests.get("https://api.openalex.org/works", params={"filter": "openalex:" + "|".join(refs[i:i + 50]), "per-page": 50, "select": "id,doi", "mailto": settings.CONTACT_EMAIL}, timeout=30).json()
        for w in j.get("results", []):
            dois[w["id"].rsplit("/", 1)[-1]] = norm_doi(w.get("doi"))
    return r["title"], set(refs), {d for d in dois.values() if d}


def evaluate(title, ks=(10, 20, 50)):
    ag = get_agents()
    real_title, ref_ids, ref_dois = survey_refs(title)
    raw = []
    for fn in (ag.search._s2, ag.search._arxiv, ag.search._crossref, ag.search._openalex):
        try:
            raw += fn(real_title, settings.PER_SOURCE)
        except Exception:
            pass
    papers = ag.search.rank(real_title, ag.search.deduplicate([dict(p) for p in raw]))
    dup = 1 - len(papers) / max(1, len(raw))
    hit = [bool(norm_doi(p.get("doi")) in ref_dois or (p.get("external_ids") or {}).get("openalex") in ref_ids) for p in papers]
    out = {"survey": real_title, "n_refs": len(ref_ids), "duplicate_rate": round(dup, 3)}
    for k in ks:
        h = hit[:k]
        dcg = sum(x / math.log2(i + 2) for i, x in enumerate(h))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(ref_ids))))
        out[f"P@{k}"], out[f"R@{k}"], out[f"NDCG@{k}"] = round(sum(h) / k, 3), round(sum(h) / max(1, len(ref_ids)), 3), round(dcg / idcg if idcg else 0, 3)
    return out


if __name__ == "__main__":
    for t in sys.argv[1:]:
        print(evaluate(t))
