import pytest
from unittest.mock import AsyncMock, patch
from src.agents.categorization import CategorizationAgent
from src.models.schemas import Paper, PaperCategory


@pytest.mark.asyncio
async def test_categorize_single_paper():
    agent = CategorizationAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Transformer for Time Series Forecasting",
        abstract="We propose a novel transformer architecture for time series forecasting.",
    )

    mock_response = '{"methodology": "deep learning", "sub_topic": "time series forecasting", "tags": ["transformer", "forecasting"]}'
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        category = await agent.categorize_paper(paper)
        assert isinstance(category, PaperCategory)
        assert category.paper_id == "arxiv:2301.00001"
        assert category.methodology == "deep learning"


@pytest.mark.asyncio
async def test_categorize_multiple_papers():
    agent = CategorizationAgent()
    papers = [
        Paper(id="a", title="Paper A", abstract="About deep learning."),
        Paper(id="b", title="Paper B", abstract="About statistical methods."),
    ]
    with patch.object(agent, "categorize_paper", new=AsyncMock(side_effect=[
        PaperCategory(paper_id="a", methodology="DL"),
        PaperCategory(paper_id="b", methodology="Stats"),
    ])):
        categories = await agent.categorize(papers)
        assert len(categories) == 2
