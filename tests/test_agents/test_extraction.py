import pytest
from unittest.mock import AsyncMock, patch
from src.agents.extraction import ExtractionAgent
from src.models.schemas import Paper, PaperExtraction


@pytest.mark.asyncio
async def test_extract_paper():
    agent = ExtractionAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Test Paper",
        abstract="We propose a new method. Experiments on ETTh1 and Weather datasets show 15% improvement.",
    )

    mock_response = '''{
        "research_question": "How to improve time series forecasting?",
        "methodology": "Modified transformer with sparse attention",
        "datasets": ["ETTh1", "Weather"],
        "key_results": "15% improvement over baselines",
        "limitations": "High computational cost"
    }'''
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        extraction = await agent.extract(paper)
        assert isinstance(extraction, PaperExtraction)
        assert extraction.paper_id == "arxiv:2301.00001"
        assert len(extraction.datasets) >= 2


@pytest.mark.asyncio
async def test_batch_extract():
    agent = ExtractionAgent()
    papers = [Paper(id="a", title="A", abstract="A."), Paper(id="b", title="B", abstract="B.")]
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value='{"datasets": [], "key_results": "test"}')):
        results = await agent.batch_extract(papers)
        assert len(results) == 2
