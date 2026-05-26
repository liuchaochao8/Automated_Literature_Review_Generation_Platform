import pytest
from unittest.mock import patch, AsyncMock
from src.services.academic_apis import (
    search_arxiv, search_semantic_scholar,
    search_crossref, deduplicate_papers
)
from src.models.schemas import Paper


@pytest.mark.asyncio
async def test_search_arxiv_mocked():
    """验证 arXiv 搜索（mock）"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.text = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001</id>
    <title>Transformer for Time Series</title>
    <summary>A paper about transformers.</summary>
    <published>2023-01-01</published>
    <author><name>John Doe</name></author>
  </entry>
</feed>"""

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        papers = await search_arxiv("transformer time series", max_results=10)
        assert len(papers) == 1
        assert papers[0].title == "Transformer for Time Series"
        assert papers[0].source == "arxiv"


@pytest.mark.asyncio
async def test_search_semantic_scholar_mocked():
    """验证 Semantic Scholar 搜索（mock）"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "data": [
            {
                "paperId": "abc123",
                "title": "Deep Learning for NLP",
                "abstract": "A comprehensive survey.",
                "year": 2023,
                "externalIds": {"ArXiv": "2301.00001"},
                "citationCount": 42,
                "authors": [{"name": "Alice"}],
            }
        ]
    }

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        papers = await search_semantic_scholar("deep learning nlp", max_results=5)
        assert len(papers) == 1
        assert papers[0].title == "Deep Learning for NLP"
        assert papers[0].source == "semantic_scholar"


@pytest.mark.asyncio
async def test_search_crossref_mocked():
    """验证 CrossRef 搜索（mock）"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test.2023.001",
                    "title": ["A Test Paper"],
                    "author": [{"given": "Bob", "family": "Smith"}],
                    "published-print": {"date-parts": [[2023]]},
                }
            ]
        }
    }

    with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_response)):
        papers = await search_crossref("test query", max_results=5)
        assert len(papers) == 1
        assert papers[0].doi == "10.1234/test.2023.001"


def test_deduplicate_by_doi():
    """验证 DOI 去重"""
    papers = [
        Paper(id="a", title="Paper A", doi="10.1234/a", source="arxiv"),
        Paper(id="b", title="Paper A", doi="10.1234/a", source="crossref"),
        Paper(id="c", title="Paper C", doi=None, source="arxiv"),
    ]
    deduped = deduplicate_papers(papers)
    assert len(deduped) == 2


def test_deduplicate_by_title():
    """验证标题模糊去重"""
    papers = [
        Paper(id="a", title="Deep Learning for Time Series Forecasting"),
        Paper(id="b", title="Deep Learning for Time Series Forecasting "),
        Paper(id="c", title="Different Paper"),
    ]
    deduped = deduplicate_papers(papers)
    assert len(deduped) == 2
