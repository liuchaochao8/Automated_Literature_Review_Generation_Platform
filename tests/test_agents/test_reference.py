import pytest
from src.agents.reference import ReferenceAgent
from src.models.schemas import Paper, Author, LiteratureReview, ReviewSection


@pytest.mark.asyncio
async def test_format_apa():
    agent = ReferenceAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Transformer for Time Series",
        authors=[Author(name="John Doe"), Author(name="Jane Smith")],
        year=2023,
        source="arxiv",
    )
    ref = agent.format_apa(paper)
    assert "Doe" in ref
    assert "2023" in ref


@pytest.mark.asyncio
async def test_format_ieee():
    agent = ReferenceAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Transformer for Time Series",
        authors=[Author(name="John Doe")],
        year=2023,
    )
    ref = agent.format_ieee(paper)
    assert "J. Doe" in ref


@pytest.mark.asyncio
async def test_attach_references():
    agent = ReferenceAgent()
    papers = [
        Paper(id="a", title="Paper A", authors=[Author(name="Alice")], year=2023),
        Paper(id="b", title="Paper B", authors=[Author(name="Bob")], year=2022),
    ]
    review = LiteratureReview(
        title="Test Review",
        sections=[ReviewSection(title="Intro", content="Content [a] and [b].")],
        references=["a", "b"],
    )
    updated = await agent.attach_references(review, papers)
    assert len(updated.references) == 2
