import logging
from typing import Any
from langgraph.graph import StateGraph, END
from src.models.state import LiteratureReviewState
from src.models.schemas import LiteratureReview, ReviewSection
from src.agents.query_expansion import QueryExpansionAgent
from src.agents.paper_discovery import PaperDiscoveryAgent
from src.agents.categorization import CategorizationAgent
from src.agents.extraction import ExtractionAgent
from src.agents.comparison import ComparisonAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.reference import ReferenceAgent
from src.agents.quality import QualityAgent

logger = logging.getLogger(__name__)


class LiteratureReviewWorkflow:
    def __init__(self):
        self.query_expansion_agent = QueryExpansionAgent()
        self.paper_discovery_agent = PaperDiscoveryAgent()
        self.categorization_agent = CategorizationAgent()
        self.extraction_agent = ExtractionAgent()
        self.comparison_agent = ComparisonAgent()
        self.synthesis_agent = SynthesisAgent()
        self.reference_agent = ReferenceAgent()
        self.quality_agent = QualityAgent()
        self.app = self.build_graph()

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(LiteratureReviewState)

        workflow.add_node("expand_queries", self._expand_queries)
        workflow.add_node("discover_papers", self._discover_papers)
        workflow.add_node("categorize_papers", self._categorize_papers)
        workflow.add_node("extract_information", self._extract_information)
        workflow.add_node("analyze_relationships", self._analyze_relationships)
        workflow.add_node("synthesize_review", self._synthesize_review)
        workflow.add_node("format_references", self._format_references)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("expand_queries")
        workflow.add_edge("expand_queries", "discover_papers")
        workflow.add_edge("discover_papers", "categorize_papers")
        workflow.add_edge("categorize_papers", "extract_information")
        workflow.add_edge("extract_information", "analyze_relationships")
        workflow.add_edge("analyze_relationships", "synthesize_review")
        workflow.add_edge("synthesize_review", "format_references")
        workflow.add_edge("format_references", "quality_check")

        workflow.add_conditional_edges(
            "quality_check",
            self._decide_next,
            {"finalize": "finalize", "synthesize_review": "synthesize_review"},
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _expand_queries(self, state: LiteratureReviewState) -> dict:
        logger.info("Expanding queries...")
        queries = await self.query_expansion_agent.expand(state["topic"])
        return {"search_queries": queries, "progress": {"expand_queries": "done"}}

    async def _discover_papers(self, state: LiteratureReviewState) -> dict:
        logger.info("Discovering papers...")
        all_papers = []
        for sq in state["search_queries"]:
            papers = await self.paper_discovery_agent.discover(sq)
            all_papers.extend(papers)

        filtered = await self.paper_discovery_agent.filter_by_relevance(all_papers, state["topic"])
        return {"discovered_papers": all_papers, "filtered_papers": filtered, "progress": {"discover_papers": "done"}}

    async def _categorize_papers(self, state: LiteratureReviewState) -> dict:
        logger.info("Categorizing papers...")
        categories = await self.categorization_agent.categorize(state["filtered_papers"])
        return {"paper_categories": categories, "progress": {"categorize_papers": "done"}}

    async def _extract_information(self, state: LiteratureReviewState) -> dict:
        logger.info("Extracting information...")
        extractions = await self.extraction_agent.batch_extract(state["filtered_papers"])
        return {"paper_extractions": extractions, "progress": {"extract_information": "done"}}

    async def _analyze_relationships(self, state: LiteratureReviewState) -> dict:
        logger.info("Analyzing relationships...")
        relations = await self.comparison_agent.analyze_all(state["filtered_papers"])
        return {"relations": relations, "progress": {"analyze_relationships": "done"}}

    async def _synthesize_review(self, state: LiteratureReviewState) -> dict:
        logger.info("Synthesizing review...")
        review = await self.synthesis_agent.synthesize(
            topic=state["topic"],
            papers=state["filtered_papers"],
            categories=state["paper_categories"],
            extractions=state["paper_extractions"],
            relations=state["relations"],
        )
        return {"review": review, "progress": {"synthesize_review": "done"}}

    async def _format_references(self, state: LiteratureReviewState) -> dict:
        logger.info("Formatting references...")
        review = await self.reference_agent.attach_references(state["review"], state["filtered_papers"])
        return {"review": review, "progress": {"format_references": "done"}}

    async def _quality_check(self, state: LiteratureReviewState) -> dict:
        logger.info("Running quality check...")
        report = await self.quality_agent.full_quality_check(state["review"], state["filtered_papers"])
        return {"quality_report": report, "progress": {"quality_check": "done"}}

    def _decide_next(self, state: LiteratureReviewState) -> str:
        if state.get("quality_report") and "critical" in state["quality_report"].lower():
            return "synthesize_review"
        return "finalize"

    async def _finalize(self, state: LiteratureReviewState) -> dict:
        logger.info("Finalizing...")
        return {"progress": {"finalize": "done"}}

    async def run(self, topic: str) -> dict[str, Any]:
        initial_state = self._make_initial_state(topic)
        result = await self.app.ainvoke(initial_state)
        return result

    async def run_stream(self, topic: str):
        """以流式方式运行工作流，每完成一个 agent 就 yield 一次状态。"""
        initial_state = self._make_initial_state(topic)
        async for update in self.app.astream(initial_state, stream_mode="updates"):
            for node_name, output in update.items():
                yield self._make_stream_event(node_name, output)

    def _make_initial_state(self, topic: str) -> LiteratureReviewState:
        return {
            "topic": topic,
            "search_queries": [],
            "discovered_papers": [],
            "filtered_papers": [],
            "paper_categories": {},
            "paper_extractions": {},
            "relations": [],
            "review": LiteratureReview(title="", sections=[]),
            "quality_report": "",
            "errors": [],
            "progress": {},
        }

    def _make_stream_event(self, node_name: str, output: dict) -> dict:
        """为每个 agent 生成前端可读的事件摘要。"""
        event = {
            "agent": node_name,
            "status": "done",
            "summary": "",
        }

        if node_name == "expand_queries":
            queries = output.get("search_queries", [])
            count = sum(len(q.expanded_queries) for q in queries)
            event["summary"] = f"生成 {count} 组检索词"

        elif node_name == "discover_papers":
            papers = output.get("filtered_papers", [])
            event["summary"] = f"发现 {len(papers)} 篇相关论文"

        elif node_name == "categorize_papers":
            cats = output.get("paper_categories", {})
            event["summary"] = f"分类 {len(cats)} 篇论文"

        elif node_name == "extract_information":
            exts = output.get("paper_extractions", {})
            event["summary"] = f"提取 {len(exts)} 篇论文信息"

        elif node_name == "analyze_relationships":
            rels = output.get("relations", [])
            event["summary"] = f"发现 {len(rels)} 组论文关系"

        elif node_name == "synthesize_review":
            review = output.get("review")
            if review:
                event["summary"] = f"生成综述: {review.title[:60]}"
            else:
                event["summary"] = "综述撰写完成"

        elif node_name == "format_references":
            event["summary"] = "引用格式化完成"

        elif node_name == "quality_check":
            event["summary"] = "质量控制检查完成"

        elif node_name == "finalize":
            event["summary"] = "综述生成完毕"
            event["status"] = "complete"

        return event
