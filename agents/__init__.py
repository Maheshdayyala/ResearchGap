"""Agent registry (singleton) shared by the Flask app and the LangGraph workflow."""
from agents.contradiction import ContradictionDetector
from agents.experiment_planning import ExperimentPlanningAgent
from agents.gap_detection import ResearchGapDetectionAgent
from agents.knowledge_graph import KnowledgeGraphBuilder
from agents.literature_search import LiteratureSearchAgent
from agents.llm import LLM
from agents.novelty import NoveltyVerificationAgent
from agents.paper_understanding import PaperUnderstandingAgent
from agents.report import ReportGenerator


class Agents:
    def __init__(self):
        self.llm = LLM()
        self.search = LiteratureSearchAgent(self.llm)
        self.understand = PaperUnderstandingAgent(self.llm)
        self.kg_builder = KnowledgeGraphBuilder(self.llm, ContradictionDetector())
        self.gap = ResearchGapDetectionAgent(self.llm)
        self.novelty = NoveltyVerificationAgent(self.search, self.llm)
        self.planner = ExperimentPlanningAgent(self.llm)
        self.report = ReportGenerator()


_agents = None


def get_agents():
    global _agents
    if _agents is None:
        _agents = Agents()
    return _agents
