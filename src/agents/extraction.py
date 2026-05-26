import json
import logging
from src.models.schemas import Paper, PaperExtraction
from src.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract key information from the following academic paper.

Title: {title}
Abstract: {abstract}

Return a JSON object with:
- "research_question": the main research question addressed
- "methodology": the proposed method/approach
- "datasets": list of datasets used
- "key_results": main experimental results
- "limitations": limitations mentioned or implied

Return ONLY valid JSON."""


class ExtractionAgent:
    def __init__(self):
        self.llm = LLMClient()

    async def extract(self, paper: Paper) -> PaperExtraction:
        prompt = EXTRACTION_PROMPT.format(title=paper.title, abstract=paper.abstract)
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at extracting structured information from academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return PaperExtraction(
                paper_id=paper.id,
                research_question=data.get("research_question"),
                methodology=data.get("methodology"),
                datasets=data.get("datasets", []),
                key_results=data.get("key_results"),
                limitations=data.get("limitations"),
            )
        except Exception as e:
            logger.warning(f"Extraction failed for {paper.id}: {e}")
            return PaperExtraction(paper_id=paper.id)

    async def batch_extract(self, papers: list[Paper]) -> dict[str, PaperExtraction]:
        import asyncio
        results = await asyncio.gather(*[self.extract(p) for p in papers], return_exceptions=True)
        extractions = {}
        for p, r in zip(papers, results):
            if isinstance(r, PaperExtraction):
                extractions[p.id] = r
            else:
                extractions[p.id] = PaperExtraction(paper_id=p.id)
        return extractions
