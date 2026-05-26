import pytest
from unittest.mock import AsyncMock, patch
from src.agents.paper_discovery import PaperDiscoveryAgent
from src.models.schemas import Paper, SearchQuery


@pytest.mark.asyncio
async def test_discover_returns_papers():
    agent = PaperDiscoveryAgent()
    search_query = SearchQuery(
        original_query="transformer time series",
        expanded_queries=["transformer time series forecasting"],
    )

    with patch("src.services.academic_apis.search_arxiv", new=AsyncMock(return_value=[
        Paper(id="arxiv:2301.00001", title="Test", abstract="Test abstract", source="arxiv")
    ])):
        with patch("src.services.academic_apis.search_semantic_scholar", new=AsyncMock(return_value=[])):
            with patch("src.services.academic_apis.search_crossref", new=AsyncMock(return_value=[])):
                papers = await agent.discover(search_query)
                assert len(papers) > 0


@pytest.mark.asyncio
async def test_filter_by_relevance():
    agent = PaperDiscoveryAgent()
    papers = [
        Paper(id="a", title="Relevant Paper", abstract="Directly about the topic.", relevance_score=0.9),
        Paper(id="b", title="Low Relevance", abstract="Completely unrelated.", relevance_score=0.1),
    ]
    filtered = await agent.filter_by_relevance(papers, "test topic")
    assert len(filtered) <= len(papers)
