import pytest
from pytest_asyncio import fixture as async_fixture
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@async_fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_review_endpoint_invalid_topic(client):
    resp = await client.post("/api/review", json={"topic": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_review_endpoint_with_mock(client):
    from unittest.mock import patch, AsyncMock
    mock_result = {
        "review": {"title": "Test Review", "sections": [], "references": [], "word_count": 100, "quality_score": None},
        "progress": {"finalize": "done"},
        "quality_report": "Quality OK",
    }
    with patch("src.orchestrator.workflow.LiteratureReviewWorkflow.run", new=AsyncMock(return_value=mock_result)):
        resp = await client.post("/api/review", json={"topic": "Transformer Time Series"})
        assert resp.status_code == 200
        data = resp.json()
        assert "review" in data
