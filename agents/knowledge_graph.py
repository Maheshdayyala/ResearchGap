"""Knowledge Graph Builder: papers / methods / datasets / topics with USES, CITES, SIMILAR_TO, CONTRADICTS edges.
The graph is built in NetworkX and mirrored to Neo4j when NEO4J_URI is configured."""
import logging
import math
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx

from agents import embeddings
from agents.contradiction import ContradictionDetector
from settings import settings
from utils import tokens

log = logging.getLogger(__name__)
LABELS = {"paper": "Paper", "method": "Method", "dataset": "Dataset", "topic": "Topic"}
RELS = {"USES_METHOD", "USES_DATASET", "CITES", "SIMILAR_TO", "CONTRADICTS", "BELONGS_TO"}


class Neo4jStore:
    def __init__(self):
        self.driver, self.ok = None, False
        if not settings.NEO4J_URI:
            return
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            self.ok = True
        except Exception as e:
            log.warning("Neo4j unavailable (%s); using in-memory NetworkX graph", e)

    def write(self, kg):
        sid = kg.search_id
        nodes, rels = defaultdict(list), defaultdict(list)
        for k, a in kg.g.nodes(data=True):
            props = {x: y for x, y in a.items() if isinstance(y, (str, int, float)) and x != "kind"}
            nodes[LABELS[a["kind"]]].append({"key": k, "props": props})
        for u, v, a in kg.g.edges(data=True):
            rels[a["rel"]].append({"s": u, "t": v, "props": {"weight": float(a.get("weight", 1.0))}})
        with self.driver.session() as s:
            s.run("MATCH (n {search_id:$sid}) DETACH DELETE n", sid=sid)
            for label, rows in nodes.items():
                s.run(f"UNWIND $rows AS r MERGE (n:{label} {{key:r.key, search_id:$sid}}) SET n += r.props", rows=rows, sid=sid)
            for rel, rows in rels.items():
                assert rel in RELS
                s.run(f"UNWIND $rows AS r MATCH (a {{key:r.s, search_id:$sid}}) MATCH (b {{key:r.t, search_id:$sid}}) "
                      f"MERGE (a)-[e:{rel}]->(b) SET e += r.props", rows=rows, sid=sid)

    def unexplored_pairs(self, sid, k, limit):
        q = """
        MATCH (m:Method {search_id:$sid})<-[:USES_METHOD]-(pm:Paper)
        WITH m, count(DISTINCT pm) AS mc WHERE mc >= $k
        MATCH (d:Dataset {search_id:$sid})<-[:USES_DATASET]-(pd:Paper)
        WITH m, mc, d, count(DISTINCT pd) AS dc
        WHERE dc >= $k AND NOT EXISTS { (m)<-[:USES_METHOD]-(:Paper)-[:USES_DATASET]->(d) }
        RETURN m.name AS method, d.name AS dataset, mc, dc ORDER BY mc*dc DESC LIMIT $limit"""
        with self.driver.session() as s:
            return [dict(r) for r in s.run(q, sid=sid, k=k, limit=limit)]


class KnowledgeGraph:
    def __init__(self, search_id, query=""):
        self.search_id = search_id
        self.query = query
        self.g = nx.MultiDiGraph()
        self.papers = {}                 # pid -> paper dict
        self.method_users = defaultdict(set)
        self.dataset_users = defaultdict(set)
        self.topics = []                 # [{id,label,pids}]
        self.store = None
        self.sim = {}

    def stats(self):
        kinds = Counter(a["kind"] for _, a in self.g.nodes(data=True))
        rels = Counter(a["rel"] for *_, a in self.g.edges(data=True))
        return {"nodes": dict(kinds), "edges": dict(rels), "backend": "neo4j+networkx" if self.store and self.store.ok else "networkx"}

    def edges_of(self, rel):
        return [(u, v, a) for u, v, a in self.g.edges(data=True) if a["rel"] == rel]

    def unexplored_pairs(self, min_use=2, limit=30):
        if self.store and self.store.ok:
            try:
                return self.store.unexplored_pairs(self.search_id, min_use, limit)
            except Exception as e:
                log.warning("Cypher query failed, falling back to NetworkX: %s", e)
        rows = []
        for m, mu in self.method_users.items():
            if len(mu) < min_use:
                continue
            for d, du in self.dataset_users.items():
                if len(du) >= min_use and not (mu & du):
                    rows.append({"method": m, "dataset": d, "mc": len(mu), "dc": len(du)})
        return sorted(rows, key=lambda r: -r["mc"] * r["dc"])[:limit]


class KnowledgeGraphBuilder:
    def __init__(self, llm=None, contradiction=None):
        self.llm = llm
        self.cd = contradiction or ContradictionDetector()
        self.store = Neo4jStore()

    def build(self, search_id, papers, persist=True, query=""):
        kg = KnowledgeGraph(search_id, query)
        kg.store = self.store
        ids = {}
        for p in papers:
            st = p.get("structured") or {}
            kg.papers[p["pid"]] = p
            pk = "paper:" + p["pid"]
            kg.g.add_node(pk, kind="paper", search_id=search_id, pid=p["pid"], title=p["title"], year=p.get("year") or 0,
                          doi=p.get("doi") or "", citations=int(p.get("citations") or 0))
            ids[(p.get("external_ids") or {}).get("openalex")] = p["pid"]
            if p.get("doi"):
                ids[p["doi"]] = p["pid"]
            for m in st.get("methods", []):
                kg.g.add_node("method:" + m, kind="method", name=m, search_id=search_id)
                kg.g.add_edge(pk, "method:" + m, rel="USES_METHOD")
                kg.method_users[m].add(p["pid"])
            for d in st.get("datasets", []):
                kg.g.add_node("dataset:" + d, kind="dataset", name=d, search_id=search_id)
                kg.g.add_edge(pk, "dataset:" + d, rel="USES_DATASET")
                kg.dataset_users[d].add(p["pid"])
        self._citations(kg, papers, ids)
        sim = self._similarity(kg, papers)
        self._topics(kg, papers, sim)
        self._contradictions(kg, papers, sim)
        if persist and self.store.ok:
            try:
                self.store.write(kg)
            except Exception as e:
                log.warning("Neo4j write failed: %s", e)
        return kg

    # ---------------------------------------------------------
    def _citations(self, kg, papers, ids):
        for p in papers:
            for r in p.get("references", []):
                tgt = ids.get(r)
                if tgt and tgt != p["pid"]:
                    kg.g.add_edge("paper:" + p["pid"], "paper:" + tgt, rel="CITES")

    def _similarity(self, kg, papers):
        if not papers:
            return None
        E = embeddings.embed([f"{p['title']}. {p['abstract'][:600]}" for p in papers])
        S = E @ E.T
        kg.sim = {(papers[i]["pid"], papers[j]["pid"]): float(S[i, j]) for i in range(len(papers)) for j in range(len(papers)) if i != j}
        added = set()
        for i, p in enumerate(papers):
            for j in S[i].argsort()[::-1][1:6]:
                key = frozenset((i, int(j)))
                if S[i, j] >= settings.SIM_EDGE and key not in added:
                    added.add(key)
                    kg.g.add_edge("paper:" + p["pid"], "paper:" + papers[j]["pid"], rel="SIMILAR_TO", weight=float(S[i, j]))
        return S

    def _topics(self, kg, papers, S):
        if not papers:
            return
        U = nx.Graph()
        U.add_nodes_from(p["pid"] for p in papers)
        for u, v, a in kg.edges_of("SIMILAR_TO"):
            U.add_edge(u[6:], v[6:], weight=a["weight"])
        try:
            comms = [set(c) for c in nx.algorithms.community.greedy_modularity_communities(U, weight="weight")] if U.number_of_edges() else [{n} for n in U.nodes]
        except Exception:
            comms = [set(c) for c in nx.connected_components(U)]
        df = Counter(w for p in papers for w in set(tokens(p["title"])))
        N = len(papers)
        titles = {p["pid"]: p["title"] for p in papers}
        for i, c in enumerate(sorted(comms, key=len, reverse=True)):
            tf = Counter(w for pid in c for w in tokens(titles[pid]))
            top = sorted(tf, key=lambda w: -tf[w] * math.log((N + 1) / (1 + df[w])))[:3]
            tid = f"topic:{i}"
            label = " ".join(top) or f"cluster {i}"
            kg.g.add_node(tid, kind="topic", label=label, search_id=kg.search_id, size=len(c))
            for pid in c:
                kg.g.add_edge("paper:" + pid, tid, rel="BELONGS_TO")
                kg.g.nodes["paper:" + pid]["topic"] = label
                kg.papers[pid]["topic"] = label
            kg.topics.append({"id": tid, "label": label, "pids": sorted(c)})

    def _contradictions(self, kg, papers, S):
        cands = []
        by = {p["pid"]: p for p in papers}
        studies = [p for p in papers if (p.get('structured') or {}).get('paper_type') != 'review']   # reviews cannot 'contradict'
        for a, b in combinations(studies, 2):
            ra, rb = (a.get("structured") or {}).get("results", ""), (b.get("structured") or {}).get("results", "")
            if not ra or not rb:
                continue
            sa, sb = a.get("structured") or {}, b.get("structured") or {}
            shared_tasks = set(sa.get("tasks", [])) & set(sb.get("tasks", []))
            shared_aspects = set(sa.get("aspects", [])) & set(sb.get("aspects", []))
            shared_datasets = set(sa.get("datasets", [])) & set(sb.get("datasets", []))
            s = kg.sim.get((a["pid"], b["pid"]), 0)
            # A shared generic method (for example, "large language model") is not enough to
            # claim contradictory evidence. Require a matching task and evaluated aspect, or
            # the same dataset with substantial paper similarity.
            same_task_facet = bool(shared_tasks and shared_aspects)
            same_dataset_context = bool(shared_datasets and s >= 0.6)
            if not (same_task_facet or same_dataset_context):
                continue
            sc = self.cd.score(ra, rb)
            if sc >= settings.CONTRADICTION_MIN:
                shared = shared_tasks | shared_aspects | shared_datasets
                cands.append((sc * (0.7 + 0.3 * s), a["pid"], b["pid"], sorted(shared)))
        for sc, x, y, shared in sorted(cands, reverse=True)[:30]:
            ok = True
            if self.llm and self.llm.available and len(cands) <= 40:
                r = self.llm.json(f'Finding A: {by[x]["structured"]["results"]}\nFinding B: {by[y]["structured"]["results"]}\n'
                                  'Do these findings genuinely contradict each other (same task/setting, opposite conclusions)? '
                                  'JSON: {"contradicts": true|false, "reason": "..."}')
                ok = bool(r.get("contradicts")) if r else True
            if ok:
                kg.g.add_edge("paper:" + x, "paper:" + y, rel="CONTRADICTS", weight=float(sc), shared=", ".join(shared))
