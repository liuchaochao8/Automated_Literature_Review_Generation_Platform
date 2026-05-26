import json
import logging
from itertools import combinations
from src.models.schemas import Paper, ComparisonRelation
from src.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """Compare the following two papers and identify relationships between them.

Paper A ({id_a}): {title_a}
Abstract: {abstract_a}

Paper B ({id_b}): {title_b}
Abstract: {abstract_b}

Identify if any of these relationships exist:
- "method_improvement": B improves upon A's method
- "method_contrast": A and B propose contrasting approaches
- "conclusion_agreement": A and B reach similar conclusions
- "conclusion_conflict": A and B have conflicting findings
- "baseline_comparison": One uses the other as baseline

Return a JSON array of relationship objects with fields: source_id, target_id, relation_type, description, confidence (0-1).
Return empty array if no clear relationship exists. ONLY valid JSON."""


class ComparisonAgent:
    def __init__(self):
        self.llm = LLMClient()

    async def compare(self, paper_a: Paper, paper_b: Paper) -> list[ComparisonRelation]:
        if paper_a.id == paper_b.id:
            return []

        prompt = COMPARISON_PROMPT.format(
            id_a=paper_a.id, title_a=paper_a.title, abstract_a=paper_a.abstract,
            id_b=paper_b.id, title_b=paper_b.title, abstract_b=paper_b.abstract,
        )
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at analyzing relationships between academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return [ComparisonRelation(**r) for r in data]
        except Exception as e:
            logger.warning(f"Comparison failed between {paper_a.id} and {paper_b.id}: {e}")
            return []

    async def analyze_all(self, papers: list[Paper]) -> list[ComparisonRelation]:
        import asyncio
        pairs = list(combinations(papers, 2))
        results = await asyncio.gather(
            *[self.compare(a, b) for a, b in pairs],
            return_exceptions=True,
        )
        all_relations = []
        for r in results:
            if isinstance(r, list):
                all_relations.extend(r)
        return all_relations
