import pytest
from unittest.mock import AsyncMock, patch
from src.agents.quality import QualityAgent
from src.models.schemas import LiteratureReview, ReviewSection, Paper


@pytest.mark.asyncio
async def test_check_consistency():
    agent = QualityAgent()
    review = LiteratureReview(
        title="Test",
        sections=[ReviewSection(title="Intro", content="Paper [a] shows 15% improvement.")],
        references=["a"],
    )
    papers = [Paper(id="a", title="A", abstract="Shows 15% improvement in accuracy.")]

    mock_response = '{"issues": [], "score": 0.95, "suggestions": ["Looks good."]}'
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        report = await agent.check_consistency(review, papers)
        assert "score" in report


def test_check_structure():
    agent = QualityAgent()
    review = LiteratureReview(
        title="Test",
        sections=[
            ReviewSection(title="Introduction", content="..."),
            ReviewSection(title="Methodology", content="..."),
            ReviewSection(title="Conclusion", content="..."),
        ],
    )
    issues = agent.check_structure(review)
    assert len(issues) == 0


def test_missing_sections():
    agent = QualityAgent()
    review = LiteratureReview(
        title="Test",
        sections=[ReviewSection(title="Intro", content="...")],
    )
    issues = agent.check_structure(review)
    assert len(issues) > 0
