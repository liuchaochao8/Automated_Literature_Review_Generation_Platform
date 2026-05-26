import asyncio
import json
import logging
from src.models.schemas import Paper, SearchQuery
from src.services.academic_apis import search_all_sources
from src.services.llm_client import LLMClient
from src.config.settings import settings

logger = logging.getLogger(__name__)


class PaperDiscoveryAgent:
    def __init__(self):
        self.llm = LLMClient()

    async def discover(self, search_query: SearchQuery, sort_by: str = "citations") -> list[Paper]:
        all_queries = search_query.expanded_queries + search_query.sub_queries
        seen_ids = set()
        papers = []

        arxiv_sort = "relevance" if sort_by == "relevance" else "submittedDate"

        for i, query in enumerate(all_queries):
            if len(papers) >= settings.max_papers:
                break
            if i > 0:
                await asyncio.sleep(0.3)  # 请求间隔，避免 API 限频
            try:
                results = await search_all_sources(query, max_results=10, arxiv_sort=arxiv_sort)
                for p in results:
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        papers.append(p)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        return self.sort_papers(papers[:settings.max_papers], sort_by)

    async def filter_by_relevance(self, papers: list[Paper], topic: str) -> list[Paper]:
        if not papers:
            return []

        prompt = f"""Rate the relevance of each paper to the topic "{topic}" on a scale of 0.0 to 1.0.
Return a JSON object mapping paper IDs to scores: {{"paper_id": score, ...}}

Papers:
{chr(10).join(f"{p.id}: {p.title}" for p in papers)}"""

        try:
            response = await self.llm.generate(
                system_prompt="You are a research assistant rating paper relevance.",
                user_prompt=prompt,
                temperature=0.0,
            )
            scores = json.loads(response)
            for p in papers:
                p.relevance_score = scores.get(p.id, 0.0)
            return [p for p in papers if p.relevance_score >= 0.3]
        except Exception as e:
            logger.warning(f"Relevance filtering failed: {e}")
            return papers

    @staticmethod
    def sort_papers(papers: list[Paper], sort_by: str = "citations") -> list[Paper]:
        if sort_by == "relevance":
            return papers

        if sort_by == "newest":
            return sorted(papers, key=lambda p: p.year or 0, reverse=True)

        # "citations": 按引用数降序，引用相同的按年份降序
        return sorted(
            papers,
            key=lambda p: (p.citations or 0, p.year or 0),
            reverse=True,
        )
