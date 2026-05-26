import pytest
from unittest.mock import AsyncMock, patch
from src.agents.query_expansion import QueryExpansionAgent
from src.models.schemas import SearchQuery


@pytest.mark.asyncio
async def test_expand_returns_search_queries():
    """验证返回 SearchQuery 列表"""
    agent = QueryExpansionAgent()
    mock_response = """
    [
        {"query": "transformer time series forecasting", "type": "method"},
        {"query": "attention mechanism temporal data", "type": "method"},
        {"query": "survey transformer time series", "type": "survey"}
    ]
    """
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        results = await agent.expand("Transformer in Time Series")
        assert len(results) > 0
        assert all(isinstance(q, SearchQuery) for q in results)


@pytest.mark.asyncio
async def test_expand_fallback_on_empty():
    """验证空结果回退到原始查询"""
    agent = QueryExpansionAgent()
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value="[]")):
        results = await agent.expand("test query")
        assert len(results) == 1
        assert results[0].original_query == "test query"
