import json
import logging
from src.models.schemas import SearchQuery
from src.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

QUERY_EXPANSION_PROMPT = """You are a literature search expert. Given a research topic, expand it into multiple search queries for academic databases.

For each query, consider different:
- Methodological angles (e.g., different technical approaches)
- Application domains
- Review/survey perspectives

Return a JSON array of objects with fields: "query" (the search query string), "type" ("method", "application", "survey", "general").

Topic: {topic}

Return ONLY valid JSON array, no other text."""


class QueryExpansionAgent:
    def __init__(self):
        self.llm = LLMClient()

    async def expand(self, topic: str) -> list[SearchQuery]:
        prompt = QUERY_EXPANSION_PROMPT.format(topic=topic)
        try:
            response = await self.llm.generate(
                system_prompt="You are a research assistant specializing in academic search strategy.",
                user_prompt=prompt,
                temperature=0.3,
            )
            queries_data = json.loads(response)
            expanded = [q["query"] for q in queries_data]

            return [SearchQuery(
                original_query=topic,
                expanded_queries=expanded,
                sub_queries=expanded,
            )]
        except Exception as e:
            logger.warning(f"Query expansion failed, using fallback: {e}")
            return [SearchQuery(
                original_query=topic,
                expanded_queries=[topic],
                sub_queries=[topic],
            )]
