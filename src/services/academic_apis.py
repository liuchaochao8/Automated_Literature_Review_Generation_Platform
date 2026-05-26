import xml.etree.ElementTree as ET
import httpx
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models.schemas import Paper, Author

logger = logging.getLogger(__name__)

ARXIV_BASE = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
CROSSREF_BASE = "https://api.crossref.org/works"

RETRY_HTTP = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
    reraise=True,
)


@retry(**RETRY_HTTP)
async def search_arxiv(query: str, max_results: int = 20) -> list[Paper]:
    """Search arXiv via OAI-PMH API."""
    params = {
        "search_query": f"all:{query}",
        "max_results": min(max_results, 100),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(ARXIV_BASE, params=params)
        resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    papers = []

    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.strip()
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published_el = entry.find("atom:published", ns)
        year = int(published_el.text[:4]) if published_el is not None else None

        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name_el = author_elem.find("atom:name", ns)
            if name_el is not None:
                authors.append(Author(name=name_el.text.strip()))

        papers.append(Paper(
            id=f"arxiv:{paper_id.split('/')[-1]}",
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            source="arxiv",
            url=paper_id,
        ))

    return papers


@retry(**RETRY_HTTP)
async def search_semantic_scholar(query: str, max_results: int = 20, api_key: Optional[str] = None) -> list[Paper]:
    """Search Semantic Scholar API."""
    from src.config.settings import settings

    headers = {}
    if api_key or settings.semantic_scholar_api_key:
        headers["x-api-key"] = api_key or settings.semantic_scholar_api_key

    params = {
        "query": query,
        "limit": min(max_results, 100),
        "fields": "title,abstract,year,authors,externalIds,citationCount,url",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params=params,
            headers=headers,
        )
        resp.raise_for_status()

    data = resp.json()
    papers = []
    for item in data.get("data", []):
        authors = [Author(name=a.get("name", "")) for a in item.get("authors", [])]
        paper_id = f"semantic_scholar:{item.get('paperId', '')}"
        ext_ids = item.get("externalIds", {})
        doi = ext_ids.get("DOI")

        papers.append(Paper(
            id=paper_id,
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            source="semantic_scholar",
            url=item.get("url"),
            doi=doi,
            citations=item.get("citationCount", 0),
        ))

    return papers


@retry(**RETRY_HTTP)
async def search_crossref(query: str, max_results: int = 20) -> list[Paper]:
    """Search CrossRef API."""
    params = {"query": query, "rows": min(max_results, 100), "sort": "relevance"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(CROSSREF_BASE, params=params)
        resp.raise_for_status()

    data = resp.json()
    papers = []
    for item in data.get("message", {}).get("items", []):
        authors = []
        for a in item.get("author", []):
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(Author(name=name))

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""
        doi = item.get("DOI", "")
        year_parts = item.get("published-print", {}).get("date-parts", [[None]])[0]
        year = year_parts[0] if year_parts else None

        papers.append(Paper(
            id=f"crossref:{doi}" if doi else f"crossref:{hash(title)}",
            title=title,
            authors=authors,
            source="crossref",
            doi=doi,
            year=year,
        ))

    return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    """模糊去重：先按 DOI，再按标题相似度。"""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    result: list[Paper] = []

    for p in papers:
        if p.doi and p.doi in seen_dois:
            continue
        if p.doi:
            seen_dois.add(p.doi)

        normalized = p.title.strip().lower().replace("  ", " ")
        if normalized in seen_titles:
            continue
        seen_titles.add(normalized)

        result.append(p)

    return result


async def search_all_sources(query: str, max_results: int = 20) -> list[Paper]:
    """并行搜索所有学术数据源。"""
    import asyncio

    tasks = [
        search_arxiv(query, max_results),
        search_semantic_scholar(query, max_results),
        search_crossref(query, max_results),
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_papers = []
    for r in results:
        if isinstance(r, Exception):
            logger.info(f"Search source unavailable (will try other sources): {r}")
            continue
        all_papers.extend(r)

    return deduplicate_papers(all_papers)
