import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.workflow import LiteratureReviewWorkflow
from src.services.academic_apis import search_arxiv
from src.models.schemas import LiteratureReview


@pytest.mark.asyncio
async def test_end_to_end_pipeline_mocked():
    """端到端流水线测试（全 mock）"""
    workflow = LiteratureReviewWorkflow()

    mock_paper = type("Paper", (), {
        "id": "test:001", "title": "Test", "abstract": "Test.",
        "authors": [], "year": 2023, "source": "arxiv",
        "url": None, "doi": None, "citations": 0,
        "relevance_score": 0.0, "keywords": [],
    })()

    with patch.multiple(
        workflow.query_expansion_agent,
        expand=AsyncMock(return_value=[type("SQ", (), {
            "original_query": "test", "expanded_queries": ["test"], "sub_queries": []
        })()]),
    ):
        with patch.multiple(
            workflow.paper_discovery_agent,
            discover=AsyncMock(return_value=[mock_paper]),
            filter_by_relevance=AsyncMock(return_value=[mock_paper]),
        ):
            with patch.object(workflow.categorization_agent, "categorize", AsyncMock(return_value={})):
                with patch.object(workflow.extraction_agent, "batch_extract", AsyncMock(return_value={})):
                    with patch.object(workflow.comparison_agent, "analyze_all", AsyncMock(return_value=[])):
                        with patch.object(workflow.synthesis_agent, "synthesize", AsyncMock(return_value=LiteratureReview(title="Test Review"))):
                            with patch.object(workflow.reference_agent, "attach_references", AsyncMock(side_effect=lambda r, p: r)):
                                with patch.object(workflow.quality_agent, "full_quality_check", AsyncMock(return_value="OK")):
                                    result = await workflow.run(topic="Test Topic")
                                    assert result is not None
                                    assert "review" in result
