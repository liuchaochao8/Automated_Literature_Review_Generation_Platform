import json
import logging
from src.models.schemas import Paper, PaperCategory
from src.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

CATEGORIZATION_PROMPT = """Analyze the following paper and categorize it.

Title: {title}
Abstract: {abstract}

Return a JSON object with these fields:
- "methodology": the main research methodology (e.g., "deep learning", "statistical", "hybrid", "theoretical")
- "time_period": approximate time period
- "research_paradigm": "theoretical", "empirical", or "survey"
- "sub_topic": specific sub-topic within the research area
- "tags": list of relevant keywords/tags

Return ONLY valid JSON."""


class CategorizationAgent:
    def __init__(self):
        self.llm = LLMClient()

    async def categorize_paper(self, paper: Paper) -> PaperCategory:
        prompt = CATEGORIZATION_PROMPT.format(title=paper.title, abstract=paper.abstract)
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at categorizing academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return PaperCategory(
                paper_id=paper.id,
                methodology=data.get("methodology"),
                time_period=data.get("time_period"),
                research_paradigm=data.get("research_paradigm"),
                sub_topic=data.get("sub_topic"),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.warning(f"Categorization failed for {paper.id}: {e}")
            return PaperCategory(paper_id=paper.id)

    async def categorize(self, papers: list[Paper]) -> dict[str, PaperCategory]:
        import asyncio
        results = await asyncio.gather(*[self.categorize_paper(p) for p in papers], return_exceptions=True)
        categories = {}
        for p, r in zip(papers, results):
            if isinstance(r, PaperCategory):
                categories[p.id] = r
            else:
                categories[p.id] = PaperCategory(paper_id=p.id)
        return categories
