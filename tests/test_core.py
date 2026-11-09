"""Offline tests: no network, no LLM. Run: pytest -q"""
import os

os.environ.update(USE_SBERT="false", DATABASE_URL="sqlite://", MIN_PAPERS="3", LLM_PROVIDER="none")

import pytest

from agents import get_agents
from agents.experiment_planning import ExperimentPlanningAgent
from agents.literature_search import LiteratureSearchAgent, new_paper
from settings import settings
from workflow import run_workflow

ICU = "intensive care unit mortality prediction"
CORPUS = [
    ("Federated learning for ICU mortality prediction", 2022, "We propose federated learning for mortality prediction on the MIMIC-III dataset. The model achieves high AUROC across hospitals. However, it is limited to a single country."),
    ("Federated learning for sepsis prediction in the ICU", 2023, "We study federated learning for sepsis prediction using the eICU database. Results show improved AUROC. Evaluation is limited to a single country."),
    ("Fairness of random forest models in ICU mortality prediction", 2021, "We evaluate fairness and bias of random forest models for mortality prediction on MIMIC-III, finding disparities across demographic subgroups."),
    ("Bias in logistic regression risk scores for ICU readmission", 2022, "We investigate bias and fairness of logistic regression readmission prediction on MIMIC-III and report disparities between groups."),
    ("A survey of fairness and privacy in clinical artificial intelligence", 2023, "This review discusses fairness, bias and privacy concerns of clinical AI, including federated learning approaches."),
    ("Privacy risks of clinical prediction: a review", 2022, "This survey reviews privacy and data leakage in clinical machine learning for ICU mortality prediction."),
]


class FakeSearch(LiteratureSearchAgent):
    def fetch_ex(self, q, per_source=None):
        ps = [new_paper(title=t, year=y, abstract=a, doi=f"10.1/{i}", citations=5, sources=["fake"]) for i, (t, y, a) in enumerate(CORPUS)]
        return self.rank(q, ps), {"fake": "ok (6)"}

    def citing_papers(self, p, limit=30):
        return []


@pytest.fixture(scope="module")
def ag():
    a = get_agents()
    a.search = FakeSearch(a.llm)
    a.novelty.search = a.search
    ExperimentPlanningAgent._hf_search = staticmethod(lambda gap: [])
    return a


def understood(ag):
    ps = ag.search.fetch(ICU)
    for p, st in zip(ps, ag.understand.extract_many(ps)):
        p["structured"] = st
    return ps


def test_html_stripped_from_abstract():
    assert "<" not in new_paper(title="t", abstract="<p><span>Hello</span> world</p>")["abstract"]


def test_extraction_is_clean(ag):
    r = ag.understand.extract(new_paper(title="Federated learning for COVID-19 MRI", abstract="We propose a deep learning, AI-based approach using federated learning on the MIMIC-III dataset. Generating datasets is hard."))
    assert "federated learning" in r["methods"] and "mimic-iii" in r["datasets"]
    assert not {"deep learning", "ai-based", "covid", "mri"} & set(r["methods"]) and "generating" not in r["datasets"]


def test_review_detection_and_aspects(ag):
    ps = {p["title"][:20]: p["structured"] for p in understood(ag)}
    assert ps["A survey of fairness"]["paper_type"] == "review" and "fairness" in ps["A survey of fairness"]["aspects"]
    assert ps["Federated learning f"]["paper_type"] == "study"


def test_hedging_sentence_is_not_a_limitation(ag):
    r = ag.understand.extract(new_paper(title="x", abstract="AI is promising. However, there are also possibilities that it may continue, or exacerbate, current disparities. It is limited to a single hospital cohort in this work."))
    assert not any("possibilities" in s for s in r["limitations"]) and r["limitations"]


def test_concept_gap_federated_x_fairness(ag):
    kg = ag.kg_builder.build(1, understood(ag), persist=False, query=ICU)
    gaps = ag.gap.detect(kg)
    hit = [g for g in gaps if g["type"] == "CONCEPT_GAP" and "federated learning" in g["components"]["methods"] and "fairness" in g["components"]["aspects"]]
    assert hit and hit[0]["missing"] and hit[0]["evidence"]
    assert all("possibilities" not in g["statement"] for g in gaps)


def _gap():
    return {"id": "x", "type": "CONCEPT_GAP", "title": "federated learning and fairness", "statement": "No study examines federated learning with respect to fairness in ICU mortality prediction.",
            "missing": "Systematic evaluation of fairness.", "evidence": [], "components": {"methods": ["federated learning"], "aspects": ["fairness"], "tasks": ["prediction"], "datasets": [], "topic": ICU}}


def test_novelty_verdicts(ag):
    corpus = understood(ag)
    assert ag.novelty.verify(_gap(), corpus, expand=False)["verdict"] == "NOVEL"
    review = new_paper(title="Federated learning and fairness: a survey in ICU mortality prediction", year=2024, abstract="This survey reviews federated learning and fairness in ICU mortality prediction.")
    assert ag.novelty.verify(_gap(), corpus + [review], expand=False)["verdict"] == "PARTIALLY_EXPLORED"
    study = new_paper(title="Fair federated learning for ICU mortality prediction", year=2024, abstract="We propose a federated learning method that improves fairness across subgroups in ICU mortality prediction on MIMIC-IV.")
    res = ag.novelty.verify(_gap(), corpus + [study], expand=False)
    assert res["verdict"] == "ALREADY_ADDRESSED" and res["closest"][0]["title"] == study["title"]


def test_plan_is_grounded(ag):
    plan = ag.planner.plan(_gap(), understood(ag))
    names = [d["name"] for d in plan["datasets"]]
    assert any(n.startswith("MIMIC") or n.startswith("eICU") for n in names) and all(d["reason"] for d in plan["datasets"])
    assert any("Reweighing" in b["name"] for b in plan["baselines"]) and not {"machine learning", "ai-based"} & {b["name"] for b in plan["baselines"]}
    assert any("parity" in m["name"].lower() or "equalized" in m["name"].lower() for m in plan["metrics"])
    assert any("fairness" in s.lower() for s in plan["methodology"])


def test_pipeline_end_to_end(ag):
    stages = []
    s = run_workflow(ag, 1, ICU, lambda st, p, m: stages.append(st))
    assert s["status"] == "done" and {"search", "understand", "graph", "gaps", "novelty", "report"} <= set(stages)
    assert s["evaluated"] and s["verified_gap"] and s["plan"]["datasets"]
    assert s["novelty"]["verdict"] in ("NOVEL", "PARTIALLY_EXPLORED") and "# Research Gap Report" in s["report_md"]


def test_dedupe_merges_same_doi():
    out = LiteratureSearchAgent().deduplicate([new_paper(title="Deep Graphs", doi="10.1/x", sources=["a"]), new_paper(title="Deep graphs!", doi="https://doi.org/10.1/X", abstract="longer abstract", sources=["b"])])
    assert len(out) == 1 and out[0]["sources"] == ["a", "b"]


def test_api_auth_flow():
    from app import create_app
    app = create_app({"TESTING": True, "WTF_CSRF_ENABLED": False, "RATELIMIT_ENABLED": False, "SQLALCHEMY_DATABASE_URI": "sqlite://"})
    c = app.test_client()
    assert c.post("/register", data={"email": "a@b.co", "password": "password1"}).status_code == 302
    assert c.post("/search", json={"query": "x"}).status_code == 400
    assert c.get("/health").json["status"] == "ok"
