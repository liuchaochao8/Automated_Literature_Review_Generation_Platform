import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.workflow import LiteratureReviewWorkflow
from src.models.schemas import (
    Paper, SearchQuery, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection,
)


@pytest.mark.asyncio
async def test_workflow_initialization():
    workflow = LiteratureReviewWorkflow()
    assert workflow.app is not None


@pytest.mark.asyncio
async def test_full_pipeline_mocked():
    workflow = LiteratureReviewWorkflow()

    workflow.query_expansion_agent.expand = AsyncMock(return_value=[
        SearchQuery(original_query="test", expanded_queries=["test query"])
    ])
    workflow.paper_discovery_agent.discover = AsyncMock(return_value=[
        Paper(id="test:001", title="Test Paper", abstract="Test.")
    ])
    workflow.paper_discovery_agent.filter_by_relevance = AsyncMock(return_value=[
        Paper(id="test:001", title="Test Paper", abstract="Test.")
    ])
    workflow.categorization_agent.categorize = AsyncMock(return_value={
        "test:001": PaperCategory(paper_id="test:001", methodology="test"),
    })
    workflow.extraction_agent.batch_extract = AsyncMock(return_value={
        "test:001": PaperExtraction(paper_id="test:001", key_results="test"),
    })
    workflow.comparison_agent.analyze_all = AsyncMock(return_value=[
        ComparisonRelation(source_id="a", target_id="b", relation_type="citation", description="", confidence=1.0),
    ])
    workflow.synthesis_agent.synthesize = AsyncMock(return_value=LiteratureReview(
        title="Test Review",
        sections=[ReviewSection(title="Intro", content="Test.")],
    ))
    workflow.reference_agent.attach_references = AsyncMock(side_effect=lambda r, p: r)
    workflow.quality_agent.full_quality_check = AsyncMock(return_value="Quality OK")

    result = await workflow.run(topic="Test Topic")
    assert "review" in result
    assert result["review"].title == "Test Review"


def test_workflow_graph_structure():
    workflow = LiteratureReviewWorkflow()
    graph = workflow.build_graph()
    assert graph is not None
