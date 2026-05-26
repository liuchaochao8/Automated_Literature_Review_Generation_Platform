import pytest
from unittest.mock import AsyncMock, patch
from src.agents.synthesis import SynthesisAgent
from src.models.schemas import (
    Paper, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection
)


@pytest.mark.asyncio
async def test_synthesize_review():
    agent = SynthesisAgent()
    papers = [Paper(id="a", title="Paper A", abstract="Method A.")]
    categories = {"a": PaperCategory(paper_id="a", methodology="deep learning")}
    extractions = {"a": PaperExtraction(paper_id="a", key_results="Good results.")}
    relations = [ComparisonRelation(source_id="b", target_id="a", relation_type="citation", description="", confidence=1.0)]

    mock_response = "# Review Title\n\n## Introduction\nTest content.\n\n## References\n[a]"
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        review = await agent.synthesize(
            topic="Test Topic",
            papers=papers,
            categories=categories,
            extractions=extractions,
            relations=relations,
        )
        assert isinstance(review, LiteratureReview)
        assert review.title


@pytest.mark.asyncio
async def test_synthesis_empty_papers():
    agent = SynthesisAgent()
    review = await agent.synthesize("test", [], {}, {}, [])
    assert isinstance(review, LiteratureReview)
