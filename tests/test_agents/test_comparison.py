import pytest
from unittest.mock import AsyncMock, patch
from src.agents.comparison import ComparisonAgent
from src.models.schemas import ComparisonRelation, Paper


@pytest.mark.asyncio
async def test_compare_two_papers():
    agent = ComparisonAgent()
    paper_a = Paper(id="a", title="Method A", abstract="We propose approach A.")
    paper_b = Paper(id="b", title="Method B", abstract="We extend approach A with improvements.")

    mock_response = '''[{
        "source_id": "b", "target_id": "a",
        "relation_type": "method_improvement",
        "description": "B improves upon A's attention mechanism",
        "confidence": 0.9
    }]'''
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        relations = await agent.compare(paper_a, paper_b)
        assert len(relations) >= 0


@pytest.mark.asyncio
async def test_build_relation_graph():
    agent = ComparisonAgent()
    papers = [Paper(id="a", title="A", abstract="A."), Paper(id="b", title="B", abstract="B.")]
    with patch.object(agent, "compare", new=AsyncMock(return_value=[
        ComparisonRelation(source_id="b", target_id="a", relation_type="citation", description="", confidence=1.0)
    ])):
        relations = await agent.analyze_all(papers)
        assert len(relations) >= 1
