import xml.etree.ElementTree as ET
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from src.models.schemas import Paper, Author

logger = logging.getLogger(__name__)

ARXIV_BASE = "https://export.arxiv.org/api/query"
OPENALEX_BASE = "https://api.openalex.org/works"
CROSSREF_BASE = "https://api.crossref.org/works"


def _should_retry(exc: BaseException) -> bool:
    """服务端错误（502/503）和超时重试；429 限频不重试，直接跳过。"""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code != 429
    return False


RETRY_HTTP = dict(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception(_should_retry),
    reraise=True,
)


@retry(**RETRY_HTTP)
async def search_arxiv(query: str, max_results: int = 20, sort_by: str = "submittedDate") -> list[Paper]:
    """Search arXiv via OAI-PMH API."""
    params = {
        "search_query": f"all:{query}",
        "max_results": min(max_results, 100),
        "sortBy": sort_by,
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "AutomatedLiteratureReview/1.0 (https://github.com/liuchaochao8/Automated_Literature_Review_Generation_Platform; mailto:)"}
    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        resp = await client.get(ARXIV_BASE, params=params, headers=headers)
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
async def search_openalex(query: str, max_results: int = 20) -> list[Paper]:
    """Search OpenAlex API（免费，不限频，含引用数）。"""
    params = {
        "search": query,
        "per_page": min(max_results, 100),
        "sort": "cited_by_count:desc",
        "select": "id,title,authorships,publication_year,cited_by_count,primary_location,doi,abstract_inverted_index",
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(OPENALEX_BASE, params=params)
        resp.raise_for_status()

    data = resp.json()
    papers = []
    for item in data.get("results", []):
        authors = []
        for a in item.get("authorships", []):
            name = a.get("author", {}).get("display_name", "")
            if name:
                authors.append(Author(name=name))

        # OpenAlex 的摘要用倒排索引编码，需要还原
        inv_index = item.get("abstract_inverted_index")
        abstract = _restore_abstract(inv_index) if inv_index else ""

        doi = item.get("doi", "").replace("https://doi.org/", "") if item.get("doi") else None

        papers.append(Paper(
            id=f"openalex:{item['id'].split('/')[-1]}",
            title=item.get("title", ""),
            authors=authors,
            abstract=abstract,
            year=item.get("publication_year"),
            source="openalex",
            doi=doi,
            citations=item.get("cited_by_count", 0),
        ))

    return papers


def _restore_abstract(inv_index: dict) -> str:
    """将 OpenAlex 的倒排索引摘要还原为文本。"""
    word_positions = []
    for word, positions in inv_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


@retry(**RETRY_HTTP)
async def search_crossref(query: str, max_results: int = 20) -> list[Paper]:
    """Search CrossRef API."""
    params = {"query": query, "rows": min(max_results, 100), "sort": "relevance"}

    async with httpx.AsyncClient(timeout=15.0) as client:
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


async def search_all_sources(query: str, max_results: int = 20, arxiv_sort: str = "submittedDate") -> list[Paper]:
    """并行搜索所有学术数据源。"""
    import asyncio

    sources = [
        ("arxiv", search_arxiv(query, max_results, sort_by=arxiv_sort)),
        ("openalex", search_openalex(query, max_results)),
        ("crossref", search_crossref(query, max_results)),
    ]

    results = await asyncio.gather(*[t[1] for t in sources], return_exceptions=True)

    all_papers = []
    for (name, _), r in zip(sources, results):
        if isinstance(r, Exception):
            reason = str(r) or type(r).__name__
            logger.info(f"Search source unavailable ({name}): {reason[:120]}")
            continue
        all_papers.extend(r)

    return deduplicate_papers(all_papers)
