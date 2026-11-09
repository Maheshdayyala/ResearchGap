"""Automated Cypher schema validation of the Neo4j knowledge graph (Table 4, Knowledge Graph Builder).
Usage: python -m evaluation.graph_validation <search_id>"""
import sys

from settings import settings

CHECKS = {
    "orphan_nodes": "MATCH (n {search_id:$sid}) WHERE NOT (n)--() RETURN count(n) AS c",
    "invalid_edge_labels": "MATCH ({search_id:$sid})-[r]->() WHERE NOT type(r) IN ['USES_METHOD','USES_DATASET','CITES','SIMILAR_TO','CONTRADICTS','BELONGS_TO'] RETURN count(r) AS c",
    "papers_missing_title": "MATCH (p:Paper {search_id:$sid}) WHERE p.title IS NULL OR p.title = '' RETURN count(p) AS c",
    "papers_missing_doi": "MATCH (p:Paper {search_id:$sid}) WHERE p.doi IS NULL OR p.doi = '' RETURN count(p) AS c",
    "methods_missing_name": "MATCH (m:Method {search_id:$sid}) WHERE m.name IS NULL OR m.name = '' RETURN count(m) AS c",
    "total_papers": "MATCH (p:Paper {search_id:$sid}) RETURN count(p) AS c",
}
SAMPLE = "MATCH (a:Paper {search_id:$sid})-[r:USES_METHOD|USES_DATASET|CONTRADICTS|CITES]->(b) WITH a,r,b,rand() AS x ORDER BY x LIMIT 20 RETURN a.title AS paper, type(r) AS rel, coalesce(b.name,b.title) AS target"

if __name__ == "__main__":
    from neo4j import GraphDatabase
    sid = int(sys.argv[1])
    with GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)).session() as s:
        for name, q in CHECKS.items():
            print(f"{name:24s}", s.run(q, sid=sid).single()["c"])
        print("\n20 random edges for manual spot-check:")
        for r in s.run(SAMPLE, sid=sid):
            print(f"  {r['paper'][:60]} -[{r['rel']}]-> {r['target'][:60]}")
