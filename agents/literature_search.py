"""Literature Search Agent: parallel retrieval from Semantic Scholar, arXiv, CrossRef, OpenAlex -> dedupe -> rank."""
import logging
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import requests

from agents import embeddings
from settings import settings
from utils import clip, norm_doi, norm_title, pid_for, strip_tags, this_year, tokens

log = logging.getLogger(__name__)
HEADERS = {"User-Agent": f"ResearchGapAI/1.0 (mailto:{settings.CONTACT_EMAIL})"}


def new_paper(**kw):
    p = {"title": "", "authors": [], "year": None, "abstract": "", "venue": "", "doi": None, "url": "",
         "citations": 0, "sources": [], "external_ids": {}, "references": [], "structured": None}
    p.update({k: v for k, v in kw.items() if v is not None})
    p["abstract"] = strip_tags(p["abstract"])
    p["title"] = strip_tags(p["title"])
    p["doi"] = norm_doi(p["doi"])
    p["pid"] = pid_for(p["doi"], p["title"])
    return p


def _get(url, params=None, headers=None, retries=2, timeout=20):
    h = dict(HEADERS)
    h.update(headers or {})
    for i in range(retries + 1):
        r = None
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r
        except Exception as e:
            if i == retries:
                raise
            time.sleep(1.5 * (i + 1))
    return None


class LiteratureSearchAgent:
    def __init__(self, llm=None):
        self.llm = llm

    # ---------------- sources ----------------
    def _s2(self, q, n):
        hdr = {"x-api-key": settings.S2_API_KEY} if settings.S2_API_KEY else {}
        d = _get("https://api.semanticscholar.org/graph/v1/paper/search", {
            "query": q, "limit": min(n, 100),
            "fields": "title,abstract,year,authors,externalIds,citationCount,venue,url"}, hdr).json()
        out = []
        for x in d.get("data", []) or []:
            ext = x.get("externalIds") or {}
            out.append(new_paper(title=x.get("title") or "", abstract=x.get("abstract") or "", year=x.get("year"),
                                 authors=[a.get("name", "") for a in x.get("authors", [])][:10],
                                 venue=x.get("venue") or "", doi=ext.get("DOI"), url=x.get("url") or "",
                                 citations=x.get("citationCount") or 0, sources=["semantic_scholar"],
                                 external_ids={"s2": x.get("paperId"), "arxiv": ext.get("ArXiv")}))
        return out

    def _arxiv(self, q, n):
        toks = tokens(q)[:5] or [q]
        r = _get("http://export.arxiv.org/api/query", {
            "search_query": " AND ".join(f"all:{t}" for t in toks), "start": 0, "max_results": min(n, 100),
            "sortBy": "relevance"})
        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            g = lambda tag: (e.findtext(f"a:{tag}", default="", namespaces=ns) or "").strip()
            url = g("id")
            out.append(new_paper(
                title=" ".join(g("title").split()), abstract=" ".join(g("summary").split()),
                year=int(g("published")[:4]) if g("published") else None,
                authors=[a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)][:10],
                venue="arXiv", doi=e.findtext("x:doi", default=None, namespaces=ns), url=url,
                sources=["arxiv"], external_ids={"arxiv": url.rsplit("/abs/", 1)[-1]}))
        return out

    def _crossref(self, q, n):
        d = _get("https://api.crossref.org/works", {
            "query": q, "rows": min(n, 100), "mailto": settings.CONTACT_EMAIL,
            "select": "DOI,title,abstract,author,issued,container-title,is-referenced-by-count,URL"}).json()
        out = []
        for x in d.get("message", {}).get("items", []):
            parts = (x.get("issued") or {}).get("date-parts") or [[None]]
            out.append(new_paper(
                title=(x.get("title") or [""])[0], abstract=strip_tags(x.get("abstract", "")),
                year=parts[0][0], doi=x.get("DOI"), url=x.get("URL", ""),
                authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in x.get("author", [])][:10],
                venue=(x.get("container-title") or [""])[0], citations=x.get("is-referenced-by-count", 0),
                sources=["crossref"]))
        return out

    def _openalex(self, q, n):
        d = _get("https://api.openalex.org/works", {
            "search": q, "per-page": min(n, 100), "filter": "has_abstract:true", "mailto": settings.CONTACT_EMAIL}).json()
        out = []
        for w in d.get("results", []):
            inv = w.get("abstract_inverted_index") or {}
            words = sorted(((i, t) for t, idx in inv.items() for i in idx))
            wid = (w.get("id") or "").rsplit("/", 1)[-1]
            out.append(new_paper(
                title=w.get("title") or "", abstract=" ".join(t for _, t in words), year=w.get("publication_year"),
                doi=w.get("doi"), url=w.get("doi") or w.get("id") or "", citations=w.get("cited_by_count", 0),
                authors=[a["author"]["display_name"] for a in w.get("authorships", [])][:10],
                venue=(((w.get("primary_location") or {}).get("source")) or {}).get("display_name") or "",
                sources=["openalex"], external_ids={"openalex": wid},
                references=[r.rsplit("/", 1)[-1] for r in w.get("referenced_works", [])]))
        return out

    # ---------------- orchestration ----------------
    def fetch_ex(self, query, per_source=None):
        """Parallel retrieval. Returns (ranked_papers, per-source status). Degrades gracefully if a source fails."""
        n = per_source or settings.PER_SOURCE
        fns = {"semantic_scholar": self._s2, "arxiv": self._arxiv, "crossref": self._crossref, "openalex": self._openalex}
        status, collected = {}, []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {name: ex.submit(fn, query, n) for name, fn in fns.items()}
            for name, f in futs.items():
                try:
                    res = f.result()
                    status[name] = f"ok ({len(res)})"
                    collected += res
                except Exception as e:
                    status[name] = f"failed: {type(e).__name__}"
                    log.warning("%s failed: %s", name, e)
        papers = [p for p in self.deduplicate(collected) if p["title"]]
        return self.rank(query, papers)[: settings.MAX_PAPERS], status

    def fetch(self, query, per_source=None):
        return self.fetch_ex(query, per_source)[0]

    def deduplicate(self, papers):
        by_doi, by_title, norms, out = {}, {}, [], []
        for p in papers:
            doi, nt = norm_doi(p.get("doi")), norm_title(p.get("title"))
            if not nt:
                continue
            tgt = by_doi.get(doi) if doi else None
            tgt = tgt or by_title.get(nt)
            if tgt is None:
                for nt2, q in norms:
                    if abs(len(nt) - len(nt2)) < 12 and SequenceMatcher(None, nt, nt2).ratio() >= 0.93:
                        tgt = q
                        break
            if tgt is None:
                out.append(p)
                norms.append((nt, p))
                by_title[nt] = p
                if doi:
                    by_doi[doi] = p
            else:
                self._merge(tgt, p)
                if doi:
                    by_doi[doi] = tgt
        return out

    @staticmethod
    def _merge(a, b):
        if len(b.get("abstract", "")) > len(a.get("abstract", "")):
            a["abstract"] = b["abstract"]
        for k in ("doi", "url", "venue", "year"):
            if not a.get(k) and b.get(k):
                a[k] = b[k]
        if not a.get("authors"):
            a["authors"] = b.get("authors", [])
        a["citations"] = max(a.get("citations", 0), b.get("citations", 0))
        a["sources"] = sorted(set(a.get("sources", [])) | set(b.get("sources", [])))
        a["references"] = sorted(set(a.get("references", [])) | set(b.get("references", [])))
        a["external_ids"] = {**{k: v for k, v in b.get("external_ids", {}).items() if v}, **{k: v for k, v in a.get("external_ids", {}).items() if v}}
        a["pid"] = pid_for(a.get("doi"), a["title"])

    def rank(self, query, papers):
        if not papers:
            return []
        E = embeddings.embed([f"{p['title']}. {p['abstract'][:600]}" for p in papers])
        rel = E @ embeddings.embed([query])[0]
        mx = max((p.get("citations") or 0) for p in papers)
        cy = this_year()
        for p, r in zip(papers, rel):
            rec = clip(1 - (cy - (p.get("year") or cy - 15)) / 15)
            cit = math.log1p(p.get("citations") or 0) / math.log1p(mx) if mx else 0
            p["relevance"] = float(r)
            p["rank_score"] = float(0.6 * clip((r + 0.1) / 0.8) + 0.2 * rec + 0.2 * cit)
        return sorted(papers, key=lambda p: -p["rank_score"])

    def refine_query(self, query, attempt=0):
        """Automatic query refinement when too few papers are returned."""
        if self.llm and self.llm.available:
            r = self.llm.json(f'Research topic query: "{query}". It returned too few results. Propose one broader or '
                              f'synonym-expanded keyword query (max 8 words). JSON: {{"query": "..."}}')
            if r and isinstance(r.get("query"), str) and r["query"].strip():
                return r["query"].strip()[:200]
        toks = tokens(query)
        options = [query + " survey", " ".join(toks[:3]) or query, " ".join(toks[:2]) or query, query + " review"]
        return options[attempt % len(options)]

    def citing_papers(self, paper, limit=30):
        """Citation-network expansion via Semantic Scholar: papers that cite `paper`."""
        ident = paper.get("external_ids", {}).get("s2") or (f"DOI:{paper['doi']}" if paper.get("doi") else None)
        if not ident:
            return []
        hdr = {"x-api-key": settings.S2_API_KEY} if settings.S2_API_KEY else {}
        try:
            d = _get(f"https://api.semanticscholar.org/graph/v1/paper/{ident}/citations", {
                "fields": "title,abstract,year,authors,externalIds,citationCount,venue", "limit": limit}, hdr).json()
        except Exception as e:
            log.warning("citation lookup failed: %s", e)
            return []
        out = []
        for it in d.get("data", []) or []:
            x = it.get("citingPaper") or {}
            ext = x.get("externalIds") or {}
            if x.get("title"):
                out.append(new_paper(title=x["title"], abstract=x.get("abstract") or "", year=x.get("year"),
                                     authors=[a.get("name", "") for a in x.get("authors", [])][:5],
                                     venue=x.get("venue") or "", doi=ext.get("DOI"),
                                     citations=x.get("citationCount") or 0, sources=["semantic_scholar"],
                                     external_ids={"s2": x.get("paperId")}))
        return out
