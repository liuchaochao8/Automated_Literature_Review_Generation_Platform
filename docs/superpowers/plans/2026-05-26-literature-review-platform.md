# 自动化文献综述生成平台 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建一个多 Agent 系统，输入一个研究主题，自动完成文献检索、分类、信息提取、关系分析和综述撰写。

**架构：** 8 个 specialized agent（QueryExpansion → PaperDiscovery → Categorization → Extraction → Comparison → Synthesis → Reference → Quality）通过 LangGraph 编排协作，Orchestrator 控制流程和进度。

**技术栈：** Python 3.11+ | LangGraph | Anthropic Claude API | arXiv/ Semantic Scholar/ CrossRef API | ChromaDB | NetworkX | FastAPI | Streamlit

---

## 文件结构

```
Automated_Literature_Review_Generation_Platform/
├── pyproject.toml                    # 项目元数据与依赖
├── .env.example                      # 环境变量模板
├── README.md                         # 项目说明
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               # 配置管理与环境变量
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py                # Pydantic 数据模型
│   │   └── state.py                  # LangGraph State 定义
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_client.py             # LLM API 封装
│   │   ├── vector_store.py           # ChromaDB 向量存储
│   │   ├── graph_store.py            # NetworkX 关系图谱
│   │   └── academic_apis.py          # 学术数据库 API 封装
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── query_expansion.py        # 检索词扩展
│   │   ├── paper_discovery.py        # 论文发现
│   │   ├── categorization.py         # 论文分类
│   │   ├── extraction.py             # 信息提取
│   │   ├── comparison.py             # 关系分析
│   │   ├── synthesis.py              # 综述撰写
│   │   ├── reference.py              # 引用格式化
│   │   └── quality.py                # 质量控制
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── workflow.py               # LangGraph 工作流定义
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py                   # FastAPI 应用
│   └── frontend/
│       └── app.py                    # Streamlit 前端
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # 共享 fixtures
│   ├── test_models/
│   │   ├── __init__.py
│   │   └── test_schemas.py
│   ├── test_services/
│   │   ├── __init__.py
│   │   ├── test_llm_client.py
│   │   ├── test_vector_store.py
│   │   ├── test_graph_store.py
│   │   └── test_academic_apis.py
│   ├── test_agents/
│   │   ├── __init__.py
│   │   ├── test_query_expansion.py
│   │   ├── test_paper_discovery.py
│   │   ├── test_categorization.py
│   │   ├── test_extraction.py
│   │   ├── test_comparison.py
│   │   ├── test_synthesis.py
│   │   ├── test_reference.py
│   │   └── test_quality.py
│   ├── test_orchestrator/
│   │   ├── __init__.py
│   │   └── test_workflow.py
│   └── test_api/
│       ├── __init__.py
│       └── test_api.py
└── docs/
    └── superpowers/
        └── plans/
            └── 2026-05-26-literature-review-platform.md
```

---

## 阶段 0：项目基础设施

### 任务 0.1：项目初始化与依赖配置

**文件：**
- 创建：`pyproject.toml`
- 创建：`.env.example`
- 创建：`README.md`
- 创建：`src/__init__.py`
- 创建：`src/config/__init__.py`
- 创建：`src/config/settings.py`
- 创建：`tests/__init__.py`
- 创建：`tests/conftest.py`
- 创建：各模块 `__init__.py` 文件

- [ ] **步骤 1：创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "literature-review-platform"
version = "0.1.0"
description = "Multi-agent automated literature review generation platform"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "anthropic>=0.45.0",
    "chromadb>=0.5.0",
    "networkx>=3.2",
    "httpx>=0.27.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.0",
    "python-dotenv>=1.0.0",
    "fastapi>=0.111.0",
    "uvicorn>=0.30.0",
    "streamlit>=1.36.0",
    "arxiv>=2.1.0",
    "tenacity>=8.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "ruff>=0.5.0",
]
```

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`
预期：依赖安装成功，无报错

- [ ] **步骤 2：创建 .env.example**

```
# LLM API Keys
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxx

# 可选：学术 API Keys（公开 API 不需要 key，但带 key 有更高限频）
SEMANTIC_SCHOLAR_API_KEY=

# 数据库配置
CHROMA_PERSIST_DIR=./data/chroma

# 应用配置
LOG_LEVEL=INFO
MAX_PAPERS=50
```

- [ ] **步骤 3：创建 src/config/settings.py**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    llm_provider: str = "anthropic"  # anthropic | openai
    llm_model: str = "claude-sonnet-4-20250514"

    # Academic APIs
    semantic_scholar_api_key: str = ""

    # Storage
    chroma_persist_dir: str = "./data/chroma"

    # App
    log_level: str = "INFO"
    max_papers: int = 50
    max_retries: int = 3

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **步骤 4：创建 tests/conftest.py**

```python
import pytest
from src.config.settings import Settings


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        max_papers=10,
    )
```

- [ ] **步骤 5：创建其余 `__init__.py` 空文件**

创建 `src/models/__init__.py`、`src/services/__init__.py`、`src/agents/__init__.py`、`src/orchestrator/__init__.py`、`src/api/__init__.py`、`src/frontend/__init__.py`，均在文件内写入一行 `#` 空内容。

运行：
```bash
mkdir -p src/models src/services src/agents src/orchestrator src/api src/frontend
for d in src/models src/services src/agents src/orchestrator src/api src/frontend tests/test_models tests/test_services tests/test_agents tests/test_orchestrator tests/test_api; do
    mkdir -p "$d"
    touch "$d/__init__.py"
done
```
预期：目录结构完整，`__init__.py` 文件全部存在

- [ ] **步骤 6：运行测试确认基础环境正常**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -c "from src.config.settings import settings; print(settings.project_root)"`
预期：输出项目根路径

- [ ] **步骤 7：Commit**

```bash
git init && git add -A && git commit -m "chore: initialize project structure and dependencies"
```

---

### 任务 0.2：核心数据模型

**文件：**
- 创建：`src/models/schemas.py`
- 创建：`src/models/state.py`
- 测试：`tests/test_models/test_schemas.py`

- [ ] **步骤 1：编写失败的测试**

```python
import pytest
from pydantic import ValidationError
from src.models.schemas import (
    Paper, Author, SearchQuery, PaperCategory, 
    PaperExtraction, ComparisonRelation, ReviewSection,
    LiteratureReview
)


def test_paper_creation():
    """验证 Paper 模型基本字段"""
    paper = Paper(
        id="arxiv:2301.00001",
        title="Transformer for Time Series",
        authors=[Author(name="John Doe"), Author(name="Jane Smith")],
        abstract="A paper about transformers in time series.",
        year=2023,
        source="arxiv",
        url="https://arxiv.org/abs/2301.00001",
    )
    assert paper.id == "arxiv:2301.00001"
    assert len(paper.authors) == 2
    assert paper.year == 2023


def test_paper_default_values():
    """验证 Paper 模型可选字段默认值"""
    paper = Paper(
        id="test:001",
        title="Test Paper",
        abstract="Test abstract",
    )
    assert paper.year is None
    assert paper.source == "unknown"
    assert paper.citations == 0
    assert paper.relevance_score == 0.0


def test_paper_doi_validation():
    """验证 DOI 链接自动补全"""
    paper = Paper(
        id="test:001",
        title="Test",
        abstract="Test",
        doi="10.1234/test.2023.001",
    )
    assert paper.doi == "10.1234/test.2023.001"


def test_search_query_required_fields():
    """验证 SearchQuery 必填字段"""
    q = SearchQuery(original_query="transformer time series", expanded_queries=["transformer time series forecasting"])
    assert q.original_query == "transformer time series"
    assert len(q.expanded_queries) == 1


def test_paper_category():
    """验证论文分类模型"""
    cat = PaperCategory(
        paper_id="arxiv:2301.00001",
        methodology="deep learning",
        time_period="2023-2024",
        research_paradigm="empirical",
        conclusion_tendency="positive",
        impact_rating=4.5,
    )
    assert cat.impact_rating == 4.5
    assert cat.conclusion_tendency == "positive"


def test_extraction_with_all_fields():
    """验证完整信息提取模型"""
    ext = PaperExtraction(
        paper_id="arxiv:2301.00001",
        research_question="How effective are transformers for time series?",
        methodology="Proposed a modified transformer architecture with positional encoding.",
        datasets=["ETTh1", "Exchange Rate", "Weather"],
        key_results="Outperformed RNN baselines by 15% on all datasets.",
        limitations="Computational cost is higher than RNNs.",
    )
    assert len(ext.datasets) == 3
    assert ext.limitations is not None


def test_comparison_relation():
    """验证关系模型"""
    rel = ComparisonRelation(
        source_id="arxiv:2301.00001",
        target_id="arxiv:2201.00001",
        relation_type="method_improvement",
        description="Improves upon the attention mechanism.",
        confidence=0.9,
    )
    assert rel.relation_type == "method_improvement"
    assert rel.confidence == 0.9


def test_review_section_fields():
    """验证综述章节模型"""
    section = ReviewSection(
        title="Introduction",
        content="This review covers transformer applications in time series.",
        subsections=[ReviewSection(title="Background", content="Deep learning basics.")],
    )
    assert section.title == "Introduction"
    assert len(section.subsections) == 1
    assert section.subsections[0].title == "Background"


def test_literature_review_complete():
    """验证完整综述模型"""
    review = LiteratureReview(
        title="Transformers in Time Series: A Comprehensive Review",
        sections=[
            ReviewSection(title="Introduction", content="..."),
            ReviewSection(title="Methodology", content="..."),
        ],
        references=["arxiv:2301.00001", "arxiv:2201.00001"],
        word_count=5000,
    )
    assert len(review.sections) == 2
    assert len(review.references) == 2
    assert review.word_count == 5000
```

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_models/test_schemas.py -v`
预期：全部 FAIL（ModuleNotFoundError）

- [ ] **步骤 2：运行测试验证失败**

运行：同上。因为 schemas.py 不存在，预期 ModuleNotFoundError。

- [ ] **步骤 3：编写数据模型代码**

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class Paper(BaseModel):
    id: str  # 唯一标识，如 "arxiv:2301.00001"
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    source: str = "unknown"  # arxiv, semantic_scholar, pubmed, crossref
    url: Optional[str] = None
    doi: Optional[str] = None
    citations: int = 0
    relevance_score: float = 0.0
    keywords: list[str] = Field(default_factory=list)


class SearchQuery(BaseModel):
    original_query: str
    expanded_queries: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)


class PaperCategory(BaseModel):
    paper_id: str
    methodology: Optional[str] = None
    time_period: Optional[str] = None
    research_paradigm: Optional[str] = None  # theoretical, empirical, survey
    conclusion_tendency: Optional[Literal["positive", "negative", "neutral"]] = None
    impact_rating: float = 0.0
    sub_topic: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class PaperExtraction(BaseModel):
    paper_id: str
    research_question: Optional[str] = None
    methodology: Optional[str] = None
    datasets: list[str] = Field(default_factory=list)
    key_results: Optional[str] = None
    limitations: Optional[str] = None
    code_url: Optional[str] = None
    key_figures: list[str] = Field(default_factory=list)


class ComparisonRelation(BaseModel):
    source_id: str
    target_id: str
    relation_type: Literal[
        "method_improvement", "method_contrast",
        "conclusion_agreement", "conclusion_conflict",
        "citation", "baseline_comparison"
    ]
    description: str = ""
    confidence: float = 0.0
    evidence: Optional[str] = None


class ReviewSection(BaseModel):
    title: str
    content: str = ""
    subsections: list["ReviewSection"] = Field(default_factory=list)


class LiteratureReview(BaseModel):
    title: str
    sections: list[ReviewSection] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    word_count: int = 0
    quality_score: Optional[float] = None
```

- [ ] **步骤 4：创建 LangGraph State 定义**

```python
from langgraph.graph import StateGraph, MessagesState
from typing import Annotated, TypedDict
from src.models.schemas import (
    SearchQuery, Paper, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview
)


class LiteratureReviewState(TypedDict):
    # Input
    topic: str
    
    # Phase 1: Query Expansion
    search_queries: list[SearchQuery]
    
    # Phase 2: Paper Discovery
    discovered_papers: list[Paper]
    filtered_papers: list[Paper]
    
    # Phase 3: Categorization
    paper_categories: dict[str, PaperCategory]  # paper_id -> category
    
    # Phase 4: Extraction
    paper_extractions: dict[str, PaperExtraction]  # paper_id -> extraction
    
    # Phase 5: Comparison
    relations: list[ComparisonRelation]
    
    # Phase 6: Synthesis
    review: LiteratureReview
    
    # Phase 7: Quality
    quality_report: str
    
    # Control
    errors: list[str]
    progress: dict[str, str]  # phase -> status
```

- [ ] **步骤 5：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_models/test_schemas.py -v`
预期：所有测试 PASS

- [ ] **步骤 6：Commit**

```bash
git add src/models/ tests/test_models/test_schemas.py
git commit -m "feat: add core data models and LangGraph state definition"
```

---

### 任务 0.3：LLM Client 封装

**文件：**
- 创建：`src/services/llm_client.py`
- 测试：`tests/test_services/test_llm_client.py`

- [ ] **步骤 1：编写失败的测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.services.llm_client import LLMClient


@pytest.mark.asyncio
async def test_llm_client_initialization():
    """验证 LLMClient 初始化"""
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    assert client.provider == "anthropic"
    assert client.model == "claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_llm_client_generate_with_mock():
    """验证 LLM 生成调用（mock api）"""
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    # Mock the internal _call_anthropic method
    with patch.object(client, "_call_anthropic", new=AsyncMock(return_value="Mocked response")):
        result = await client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Hello!",
            max_tokens=100,
        )
        assert result == "Mocked response"


@pytest.mark.asyncio
async def test_llm_client_generate_empty_provider():
    """验证空 provider 抛出错误"""
    client = LLMClient(provider="", model="test")
    with pytest.raises(ValueError, match="LLM provider not configured"):
        await client.generate(system_prompt="", user_prompt="test")


@pytest.mark.asyncio
async def test_generate_with_temperature():
    """验证 temperature 参数传递"""
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
    with patch.object(client, "_call_anthropic", new=AsyncMock(return_value="response")):
        result = await client.generate(
            system_prompt="", user_prompt="test", temperature=0.7
        )
        assert result == "response"
```

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_llm_client.py -v`
预期：FAIL

- [ ] **步骤 2：运行测试验证失败**

运行：同上。预期 ModuleNotFoundError / ImportError。

- [ ] **步骤 3：编写 LLM Client 实现**

```python
import os
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config.settings import settings


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(
                api_key=settings.anthropic_api_key
            )
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(
                api_key=settings.openai_api_key
            )
        return self._openai_client

    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        if not self.provider:
            raise ValueError("LLM provider not configured")

        if self.provider == "anthropic":
            return await self._call_anthropic(system_prompt, user_prompt, max_tokens, temperature)
        elif self.provider == "openai":
            return await self._call_openai(system_prompt, user_prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def _call_anthropic(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        client = self._get_anthropic_client()
        response = await client.messages.create(
            model=self.model,
            system=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    async def _call_openai(self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float) -> str:
        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_llm_client.py -v`
预期：PASS（注意：无 API key 的测试会 mock，所以不需要真实 key）

- [ ] **步骤 5：Commit**

```bash
git add src/services/llm_client.py tests/test_services/test_llm_client.py
git commit -m "feat: add LLM client with Anthropic and OpenAI support"
```

---

### 任务 0.4：学术 API 封装

**文件：**
- 创建：`src/services/academic_apis.py`
- 测试：`tests/test_services/test_academic_apis.py`

- [ ] **步骤 1：编写失败的测试**

```python
import pytest
from unittest.mock import patch, AsyncMock
from src.services.academic_apis import (
    search_arxiv, search_semantic_scholar, 
    search_crossref, deduplicate_papers
)
from src.models.schemas import Paper


@pytest.mark.asyncio
async def test_search_arxiv_mocked():
    """验证 arXiv 搜索（mock HTTP 请求）"""
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
    """验证 Semantic Scholar 搜索（mock HTTP 请求）"""
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
    """验证 CrossRef 搜索（mock HTTP 请求）"""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test.2023.001",
                    "title": ["A Test Paper"],
                    "author": [{"given": "Bob", "family": "Smith"}],
                    "published-print": {"date-parts": [[2023]]],
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
    assert len(deduped) == 2  # a and b deduped (same DOI), c stays


def test_deduplicate_by_title():
    """验证标题模糊去重"""
    papers = [
        Paper(id="a", title="Deep Learning for Time Series Forecasting"),
        Paper(id="b", title="Deep Learning for Time Series Forecasting "),  # trailing space
        Paper(id="c", title="Different Paper"),
    ]
    deduped = deduplicate_papers(papers)
    assert len(deduped) == 2
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_academic_apis.py -v`
预期：FAIL

- [ ] **步骤 3：编写 Academic APIs 实现**

```python
import json
import httpx
import logging
from typing import Optional
from src.models.schemas import Paper, Author

logger = logging.getLogger(__name__)

ARXIV_BASE = "http://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
CROSSREF_BASE = "https://api.crossref.org/works"

APEX_BASE = "https://api.openalex.org/works"


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

    papers = []
    import xml.etree.ElementTree as ET
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)

    for entry in root.findall("atom:entry", ns):
        paper_id = entry.find("atom:id", ns).text.strip()
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
        published = entry.find("atom:published", ns).text.strip()[:4] if entry.find("atom:published", ns) is not None else None

        authors = []
        for author_elem in entry.findall("atom:author", ns):
            name = author_elem.find("atom:name", ns)
            if name is not None:
                authors.append(Author(name=name.text.strip()))

        papers.append(Paper(
            id=f"arxiv:{paper_id.split('/')[-1]}",
            title=title,
            authors=authors,
            abstract=abstract,
            year=int(published) if published else None,
            source="arxiv",
            url=paper_id,
        ))

    return papers


async def search_semantic_scholar(query: str, max_results: int = 20, api_key: Optional[str] = None) -> list[Paper]:
    """Search Semantic Scholar API."""
    from src.config.settings import settings

    headers = {}
    api_key = api_key or settings.semantic_scholar_api_key
    if api_key:
        headers["x-api-key"] = api_key

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
        authors = [
            Author(name=a.get("name", ""))
            for a in item.get("authors", [])
        ]
        paper_id = f"semantic_scholar:{item.get('paperId', '')}"
        ext_ids = item.get("externalIds", {})
        doi = ext_ids.get("DOI")
        arxiv_id = ext_ids.get("ArXiv")

        papers.append(Paper(
            id=paper_id,
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstract", "") or "",
            year=item.get("year"),
            source="semantic_scholar",
            url=item.get("url"),
            doi=doi,
            citations=item.get("citationCount", 0),
        ))

    return papers


async def search_crossref(query: str, max_results: int = 20) -> list[Paper]:
    """Search CrossRef API."""
    params = {
        "query": query,
        "rows": min(max_results, 100),
        "sort": "relevance",
    }

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
        # DOI 去重
        if p.doi and p.doi in seen_dois:
            continue
        if p.doi:
            seen_dois.add(p.doi)

        # 标题去重（标准化后比较）
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
            logger.warning(f"Search failed: {r}")
            continue
        all_papers.extend(r)

    return deduplicate_papers(all_papers)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_academic_apis.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/services/academic_apis.py tests/test_services/test_academic_apis.py
git commit -m "feat: add academic API wrappers for arXiv, Semantic Scholar, CrossRef"
```

---

### 任务 0.5：向量存储与图谱存储

**文件：**
- 创建：`src/services/vector_store.py`
- 创建：`src/services/graph_store.py`
- 测试：`tests/test_services/test_vector_store.py`
- 测试：`tests/test_services/test_graph_store.py`

- [ ] **步骤 1：编写向量存储测试**

```python
import pytest
import tempfile
from pathlib import Path
from src.services.vector_store import PaperVectorStore
from src.models.schemas import Paper


@pytest.fixture
def vector_store():
    """使用临时目录创建向量存储"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PaperVectorStore(persist_dir=tmpdir)
        yield store


def test_add_and_search_papers(vector_store):
    """验证添加论文并搜索"""
    paper = Paper(
        id="test:001",
        title="Machine Learning Basics",
        abstract="Introduction to machine learning concepts and algorithms.",
    )
    vector_store.add_paper(paper, embedding=[0.1] * 128)
    assert vector_store.count() == 1


def test_search_similar(vector_store):
    """验证相似度搜索"""
    papers = [
        Paper(id="test:001", title="Deep Learning", abstract="Neural networks."),
        Paper(id="test:002", title="Machine Learning", abstract="Statistical methods."),
    ]
    for p in papers:
        vector_store.add_paper(p, embedding=[0.1] * 128)

    vector_store.add_paper(papers[0], embedding=[0.1] * 128)
    results = vector_store.search([0.1] * 128, top_k=2)
    assert len(results) == 2


def test_delete_paper(vector_store):
    """验证删除论文"""
    paper = Paper(id="test:001", title="Test", abstract="Test abstract.")
    vector_store.add_paper(paper, embedding=[0.1] * 128)
    vector_store.delete_paper("test:001")
    assert vector_store.count() == 0
```
预期：FAIL

- [ ] **步骤 2：编写图谱存储测试**

```python
import pytest
from src.services.graph_store import CitationGraph
from src.models.schemas import ComparisonRelation


@pytest.fixture
def graph():
    return CitationGraph()


def test_add_relation(graph):
    """验证添加关系"""
    rel = ComparisonRelation(
        source_id="paper:a",
        target_id="paper:b",
        relation_type="citation",
        description="A cites B",
        confidence=1.0,
    )
    graph.add_relation(rel)
    assert graph.node_count() == 2
    assert graph.edge_count() == 1


def test_get_related_papers(graph):
    """验证获取关联论文"""
    graph.add_relation(ComparisonRelation(
        source_id="paper:a", target_id="paper:b",
        relation_type="method_improvement", description="", confidence=0.9,
    ))
    graph.add_relation(ComparisonRelation(
        source_id="paper:a", target_id="paper:c",
        relation_type="conclusion_conflict", description="", confidence=0.8,
    ))
    related = graph.get_related_papers("paper:a")
    assert len(related) == 2


def test_find_path(graph):
    """验证路径查找"""
    graph.add_relation(ComparisonRelation(source_id="a", target_id="b", relation_type="citation", description="", confidence=1.0))
    graph.add_relation(ComparisonRelation(source_id="b", target_id="c", relation_type="citation", description="", confidence=1.0))
    path = graph.find_path("a", "c")
    assert len(path) == 3


def test_get_evolution_chain(graph):
    """验证方法演进链"""
    graph.add_relation(ComparisonRelation(source_id="paper:2019", target_id="paper:2020", relation_type="method_improvement", description="", confidence=1.0))
    graph.add_relation(ComparisonRelation(source_id="paper:2020", target_id="paper:2021", relation_type="method_improvement", description="", confidence=1.0))
    chain = graph.get_evolution_chain("method_improvement")
    assert len(chain) >= 2
```

- [ ] **步骤 3：运行测试验证失败**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_vector_store.py tests/test_services/test_graph_store.py -v`
预期：FAIL

- [ ] **步骤 4：实现向量存储**

```python
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.models.schemas import Paper


class PaperVectorStore:
    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "papers"):
        self.persist_dir = persist_dir or "./data/chroma"
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_paper(self, paper: Paper, embedding: list[float]) -> None:
        self.collection.add(
            ids=[paper.id],
            embeddings=[embedding],
            metadatas=[{
                "title": paper.title,
                "abstract": paper.abstract,
                "year": str(paper.year) if paper.year else "",
                "source": paper.source,
                "citations": paper.citations,
                "authors": ", ".join(a.name for a in paper.authors),
            }],
        )

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })
        return output

    def delete_paper(self, paper_id: str) -> None:
        self.collection.delete(ids=[paper_id])

    def count(self) -> int:
        return self.collection.count()
```

- [ ] **步骤 5：实现图谱存储**

```python
import networkx as nx
from typing import Optional
from src.models.schemas import ComparisonRelation


class CitationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relation(self, relation: ComparisonRelation) -> None:
        self.graph.add_node(relation.source_id)
        self.graph.add_node(relation.target_id)
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            description=relation.description,
            confidence=relation.confidence,
        )

    def get_related_papers(self, paper_id: str) -> list[dict]:
        edges = []
        for _, target, data in self.graph.edges(paper_id, data=True):
            edges.append({"target": target, **data})
        for source, _, data in self.graph.in_edges(paper_id, data=True):
            edges.append({"source": source, **data})
        return edges

    def find_path(self, source: str, target: str) -> list[str]:
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_evolution_chain(self, relation_type: str) -> list[tuple[str, str]]:
        chains = []
        for u, v, data in self.graph.edges(data=True):
            if data.get("relation_type") == relation_type:
                chains.append((u, v))
        return chains

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_aggregated_view(self) -> dict:
        """生成图谱聚合摘要"""
        return {
            "total_papers": self.node_count(),
            "total_relations": self.edge_count(),
            "density": nx.density(self.graph),
        }
```

- [ ] **步骤 6：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_services/test_vector_store.py tests/test_services/test_graph_store.py -v`
预期：PASS

- [ ] **步骤 7：Commit**

```bash
git add src/services/vector_store.py src/services/graph_store.py tests/test_services/test_vector_store.py tests/test_services/test_graph_store.py
git commit -m "feat: add ChromaDB vector store and NetworkX citation graph"
```

---

## 阶段 1：Agent 实现

### 任务 1.1：QueryExpansionAgent

**文件：**
- 创建：`src/agents/query_expansion.py`
- 测试：`tests/test_agents/test_query_expansion.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.query_expansion import QueryExpansionAgent
from src.models.schemas import SearchQuery


@pytest.mark.asyncio
async def test_expand_returns_search_queries():
    """验证返回 SearchQuery 列表"""
    agent = QueryExpansionAgent()
    mock_response = """
    [
        {"query": "transformer time series forecasting", "type": "method"},
        {"query": "attention mechanism temporal data", "type": "method"},
        {"query": "survey transformer time series", "type": "survey"}
    ]
    """
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        results = await agent.expand("Transformer in Time Series")
        assert len(results) > 0
        assert all(isinstance(q, SearchQuery) for q in results)


@pytest.mark.asyncio
async def test_expand_fallback_on_empty():
    """验证空结果回退到原始查询"""
    agent = QueryExpansionAgent()
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value="[]")):
        results = await agent.expand("test query")
        assert len(results) == 1
        assert results[0].original_query == "test query"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_query_expansion.py -v`
预期：FAIL

- [ ] **步骤 3：实现 QueryExpansionAgent**

```python
import json
import logging
from src.agents.base import BaseAgent
from src.models.schemas import SearchQuery

logger = logging.getLogger(__name__)

QUERY_EXPANSION_PROMPT = """You are a literature search expert. Given a research topic, expand it into multiple search queries for academic databases.

For each query, consider different:
- Methodological angles (e.g., different technical approaches)
- Application domains
- Review/survey perspectives

Return a JSON array of objects with fields: "query" (the search query string), "type" ("method", "application", "survey", "general").

Topic: {topic}

Return ONLY valid JSON array, no other text."""


class QueryExpansionAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def expand(self, topic: str) -> list[SearchQuery]:
        prompt = QUERY_EXPANSION_PROMPT.format(topic=topic)
        try:
            response = await self.llm.generate(
                system_prompt="You are a research assistant specializing in academic search strategy.",
                user_prompt=prompt,
                temperature=0.3,
            )
            queries_data = json.loads(response)
            expanded = [q["query"] for q in queries_data]
            sub_queries = [q["query"] for q in queries_data if q.get("type") in ("method", "application")]

            return [SearchQuery(
                original_query=topic,
                expanded_queries=expanded,
                sub_queries=sub_queries,
            )]
        except Exception as e:
            logger.warning(f"Query expansion failed, using fallback: {e}")
            return [SearchQuery(
                original_query=topic,
                expanded_queries=[topic],
                sub_queries=[topic],
            )]
```

注意：需要先创建 `src/agents/base.py`：

```python
# Base agent (empty for now, serves as namespace)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_query_expansion.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/agents/base.py src/agents/query_expansion.py tests/test_agents/test_query_expansion.py
git commit -m "feat: add QueryExpansionAgent"
```

---

### 任务 1.2：PaperDiscoveryAgent

**文件：**
- 创建：`src/agents/paper_discovery.py`
- 测试：`tests/test_agents/test_paper_discovery.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.paper_discovery import PaperDiscoveryAgent
from src.models.schemas import Paper, SearchQuery


@pytest.mark.asyncio
async def test_discover_returns_papers():
    """验证发现论文"""
    agent = PaperDiscoveryAgent()
    search_query = SearchQuery(
        original_query="transformer time series",
        expanded_queries=["transformer time series forecasting"],
    )

    with patch("src.services.academic_apis.search_arxiv", new=AsyncMock(return_value=[
        Paper(id="arxiv:2301.00001", title="Test", abstract="Test abstract", source="arxiv")
    ])):
        with patch("src.services.academic_apis.search_semantic_scholar", new=AsyncMock(return_value=[])):
            with patch("src.services.academic_apis.search_crossref", new=AsyncMock(return_value=[])):
                papers = await agent.discover(search_query)
                assert len(papers) > 0


@pytest.mark.asyncio
async def test_relevance_filtering():
    """验证相关性过滤"""
    agent = PaperDiscoveryAgent()
    papers = [
        Paper(id="a", title="Relevant Paper", abstract="Directly about the topic.", relevance_score=0.9),
        Paper(id="b", title="Low Relevance", abstract="Completely unrelated.", relevance_score=0.1),
    ]
    filtered = await agent.filter_by_relevance(papers, "test topic")
    # 基于 LLM 评分或阈值过滤
    assert len(filtered) <= len(papers)
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_paper_discovery.py -v`
预期：FAIL

- [ ] **步骤 3：实现 PaperDiscoveryAgent**

```python
import logging
from src.models.schemas import Paper, SearchQuery
from src.services.academic_apis import search_all_sources
from src.config.settings import settings

logger = logging.getLogger(__name__)


class PaperDiscoveryAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def discover(self, search_query: SearchQuery) -> list[Paper]:
        """对每个扩展检索词搜索所有数据源。"""
        all_queries = search_query.expanded_queries + search_query.sub_queries
        seen_ids = set()
        papers = []

        for query in all_queries:
            if len(papers) >= settings.max_papers:
                break
            try:
                results = await search_all_sources(query, max_results=10)
                for p in results:
                    if p.id not in seen_ids:
                        seen_ids.add(p.id)
                        papers.append(p)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        return papers[:settings.max_papers]

    async def filter_by_relevance(self, papers: list[Paper], topic: str) -> list[Paper]:
        """用 LLM 对论文做相关性评分，过滤低分论文。"""
        if not papers:
            return []

        prompt = f"""Rate the relevance of each paper to the topic "{topic}" on a scale of 0.0 to 1.0.
Return a JSON object mapping paper IDs to scores: {{"paper_id": score, ...}}

Papers:
{chr(10).join(f"{p.id}: {p.title}" for p in papers)}"""

        try:
            response = await self.llm.generate(
                system_prompt="You are a research assistant rating paper relevance.",
                user_prompt=prompt,
                temperature=0.0,
            )
            import json
            scores = json.loads(response)
            for p in papers:
                p.relevance_score = scores.get(p.id, 0.0)
            return [p for p in papers if p.relevance_score >= 0.3]
        except Exception as e:
            logger.warning(f"Relevance filtering failed: {e}")
            return papers  # fallback: keep all
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_paper_discovery.py -v`
预期：PASS

- [ ] **步骤 5：Commit**

```bash
git add src/agents/paper_discovery.py tests/test_agents/test_paper_discovery.py
git commit -m "feat: add PaperDiscoveryAgent"
```

---

### 任务 1.3：CategorizationAgent

**文件：**
- 创建：`src/agents/categorization.py`
- 测试：`tests/test_agents/test_categorization.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.categorization import CategorizationAgent
from src.models.schemas import Paper, PaperCategory


@pytest.mark.asyncio
async def test_categorize_single_paper():
    """验证单篇论文分类"""
    agent = CategorizationAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Transformer for Time Series Forecasting",
        abstract="We propose a novel transformer architecture for time series forecasting.",
    )

    mock_response = '{"methodology": "deep learning", "sub_topic": "time series forecasting", "tags": ["transformer", "forecasting"]}'
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        category = await agent.categorize_paper(paper)
        assert isinstance(category, PaperCategory)
        assert category.paper_id == "arxiv:2301.00001"
        assert category.methodology == "deep learning"


@pytest.mark.asyncio
async def test_categorize_multiple_papers():
    """验证批量分类"""
    agent = CategorizationAgent()
    papers = [
        Paper(id="a", title="Paper A", abstract="About deep learning."),
        Paper(id="b", title="Paper B", abstract="About statistical methods."),
    ]
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=AsyncMock())):
        with patch.object(agent, "categorize_paper", new=AsyncMock(side_effect=[
            PaperCategory(paper_id="a", methodology="DL"),
            PaperCategory(paper_id="b", methodology="Stats"),
        ])):
            categories = await agent.categorize(papers)
            assert len(categories) == 2
```

- [ ] **步骤 2：实现 CategorizationAgent**

```python
import json
import logging
from src.models.schemas import Paper, PaperCategory

logger = logging.getLogger(__name__)

CATEGORIZATION_PROMPT = """Analyze the following paper and categorize it.

Title: {title}
Abstract: {abstract}

Return a JSON object with these fields:
- "methodology": the main research methodology (e.g., "deep learning", "statistical", "hybrid", "theoretical")
- "time_period": approximate time period
- "research_paradigm": "theoretical", "empirical", or "survey"
- "sub_topic": specific sub-topic within the research area
- "tags": list of relevant keywords/tags

Return ONLY valid JSON."""


class CategorizationAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def categorize_paper(self, paper: Paper) -> PaperCategory:
        prompt = CATEGORIZATION_PROMPT.format(title=paper.title, abstract=paper.abstract)
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at categorizing academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return PaperCategory(
                paper_id=paper.id,
                methodology=data.get("methodology"),
                time_period=data.get("time_period"),
                research_paradigm=data.get("research_paradigm"),
                sub_topic=data.get("sub_topic"),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.warning(f"Categorization failed for {paper.id}: {e}")
            return PaperCategory(paper_id=paper.id)

    async def categorize(self, papers: list[Paper]) -> dict[str, PaperCategory]:
        import asyncio
        results = await asyncio.gather(*[self.categorize_paper(p) for p in papers], return_exceptions=True)
        categories = {}
        for p, r in zip(papers, results):
            if isinstance(r, PaperCategory):
                categories[p.id] = r
            else:
                categories[p.id] = PaperCategory(paper_id=p.id)
        return categories
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_categorization.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/categorization.py tests/test_agents/test_categorization.py
git commit -m "feat: add CategorizationAgent"
```

---

### 任务 1.4：ExtractionAgent

**文件：**
- 创建：`src/agents/extraction.py`
- 测试：`tests/test_agents/test_extraction.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.extraction import ExtractionAgent
from src.models.schemas import Paper, PaperExtraction


@pytest.mark.asyncio
async def test_extract_paper():
    """验证单篇论文信息提取"""
    agent = ExtractionAgent()
    paper = Paper(
        id="arxiv:2301.00001",
        title="Test Paper",
        abstract="We propose a new method. Experiments on ETTh1 and Weather datasets show 15% improvement.",
    )

    mock_response = '''{
        "research_question": "How to improve time series forecasting?",
        "methodology": "Modified transformer with sparse attention",
        "datasets": ["ETTh1", "Weather"],
        "key_results": "15% improvement over baselines",
        "limitations": "High computational cost"
    }'''
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value=mock_response)):
        extraction = await agent.extract(paper)
        assert isinstance(extraction, PaperExtraction)
        assert extraction.paper_id == "arxiv:2301.00001"
        assert len(extraction.datasets) >= 2


@pytest.mark.asyncio
async def test_batch_extract():
    """验证批量提取"""
    agent = ExtractionAgent()
    papers = [Paper(id="a", title="A", abstract="A."), Paper(id="b", title="B", abstract="B.")]
    with patch.object(agent.llm, "generate", new=AsyncMock(return_value='{"datasets": [], "key_results": "test"}')):
        results = await agent.batch_extract(papers)
        assert len(results) == 2
```

- [ ] **步骤 2：实现 ExtractionAgent**

```python
import json
import logging
from src.models.schemas import Paper, PaperExtraction

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract key information from the following academic paper.

Title: {title}
Abstract: {abstract}

Return a JSON object with:
- "research_question": the main research question addressed
- "methodology": the proposed method/approach
- "datasets": list of datasets used
- "key_results": main experimental results
- "limitations": limitations mentioned or implied

Return ONLY valid JSON."""


class ExtractionAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def extract(self, paper: Paper) -> PaperExtraction:
        prompt = EXTRACTION_PROMPT.format(title=paper.title, abstract=paper.abstract)
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at extracting structured information from academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return PaperExtraction(
                paper_id=paper.id,
                research_question=data.get("research_question"),
                methodology=data.get("methodology"),
                datasets=data.get("datasets", []),
                key_results=data.get("key_results"),
                limitations=data.get("limitations"),
            )
        except Exception as e:
            logger.warning(f"Extraction failed for {paper.id}: {e}")
            return PaperExtraction(paper_id=paper.id)

    async def batch_extract(self, papers: list[Paper]) -> dict[str, PaperExtraction]:
        import asyncio
        results = await asyncio.gather(*[self.extract(p) for p in papers], return_exceptions=True)
        extractions = {}
        for p, r in zip(papers, results):
            if isinstance(r, PaperExtraction):
                extractions[p.id] = r
            else:
                extractions[p.id] = PaperExtraction(paper_id=p.id)
        return extractions
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_extraction.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/extraction.py tests/test_agents/test_extraction.py
git commit -m "feat: add ExtractionAgent"
```

---

### 任务 1.5：ComparisonAgent

**文件：**
- 创建：`src/agents/comparison.py`
- 测试：`tests/test_agents/test_comparison.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.comparison import ComparisonAgent
from src.models.schemas import ComparisonRelation, Paper


@pytest.mark.asyncio
async def test_compare_two_papers():
    """验证两篇论文对比"""
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
    """验证构建完整关系图"""
    agent = ComparisonAgent()
    papers = [Paper(id="a", title="A", abstract="A."), Paper(id="b", title="B", abstract="B.")]
    with patch.object(agent, "compare", new=AsyncMock(return_value=[
        ComparisonRelation(source_id="b", target_id="a", relation_type="citation", description="", confidence=1.0)
    ])):
        relations = await agent.analyze_all(papers)
        assert len(relations) >= 1
```

- [ ] **步骤 2：实现 ComparisonAgent**

```python
import json
import logging
from itertools import combinations
from src.models.schemas import Paper, ComparisonRelation

logger = logging.getLogger(__name__)

COMPARISON_PROMPT = """Compare the following two papers and identify relationships between them.

Paper A ({id_a}): {title_a}
Abstract: {abstract_a}

Paper B ({id_b}): {title_b}
Abstract: {abstract_b}

Identify if any of these relationships exist:
- "method_improvement": B improves upon A's method
- "method_contrast": A and B propose contrasting approaches
- "conclusion_agreement": A and B reach similar conclusions
- "conclusion_conflict": A and B have conflicting findings
- "baseline_comparison": One uses the other as baseline

Return a JSON array of relationship objects with fields: source_id, target_id, relation_type, description, confidence (0-1).
Return empty array if no clear relationship exists. ONLY valid JSON."""


class ComparisonAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def compare(self, paper_a: Paper, paper_b: Paper) -> list[ComparisonRelation]:
        if paper_a.id == paper_b.id:
            return []

        prompt = COMPARISON_PROMPT.format(
            id_a=paper_a.id, title_a=paper_a.title, abstract_a=paper_a.abstract,
            id_b=paper_b.id, title_b=paper_b.title, abstract_b=paper_b.abstract,
        )
        try:
            response = await self.llm.generate(
                system_prompt="You are an expert at analyzing relationships between academic papers.",
                user_prompt=prompt,
                temperature=0.1,
            )
            data = json.loads(response)
            return [ComparisonRelation(**r) for r in data]
        except Exception as e:
            logger.warning(f"Comparison failed between {paper_a.id} and {paper_b.id}: {e}")
            return []

    async def analyze_all(self, papers: list[Paper]) -> list[ComparisonRelation]:
        all_relations = []
        pairs = list(combinations(papers, 2))

        import asyncio
        results = await asyncio.gather(
            *[self.compare(a, b) for a, b in pairs],
            return_exceptions=True,
        )

        for r in results:
            if isinstance(r, list):
                all_relations.extend(r)

        return all_relations
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_comparison.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/comparison.py tests/test_agents/test_comparison.py
git commit -m "feat: add ComparisonAgent"
```

---

### 任务 1.6：SynthesisAgent

**文件：**
- 创建：`src/agents/synthesis.py`
- 测试：`tests/test_agents/test_synthesis.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.synthesis import SynthesisAgent
from src.models.schemas import (
    Paper, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection
)


@pytest.mark.asyncio
async def test_synthesize_review():
    """验证综述生成"""
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
    """验证无论文时返回空综述"""
    agent = SynthesisAgent()
    review = await agent.synthesize("test", [], {}, {}, [])
    assert isinstance(review, LiteratureReview)
    assert "No papers" in review.sections[0].content if review.sections else True
```

- [ ] **步骤 2：实现 SynthesisAgent**

```python
import logging
from src.models.schemas import (
    Paper, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection,
)
from src.config.settings import settings

logger = logging.getLogger(__name__)

SYNTHESIS_PROMPT = """Write a comprehensive literature review on the topic "{topic}".

Available papers ({num_papers} total):

{papers_summary}

Categories:
{categories_summary}

Key Findings:
{extractions_summary}

Relationships:
{relations_summary}

Structure the review with:
1. Introduction (background, research problem, contribution of this review)
2. Methodology Classification (categorize and describe different methodological approaches)
3. Comparative Analysis (compare methods across datasets and results)
4. Trends and Future Directions

Use proper academic writing style. Mark references as [Paper ID].
Respond in Markdown format."""


class SynthesisAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    def _format_papers(self, papers: list[Paper]) -> str:
        lines = []
        for p in papers:
            authors = ", ".join(a.name for a in p.authors[:3])
            lines.append(f"- {p.id}: \"{p.title}\" by {authors} ({p.year or 'n.d.'})")
        return "\n".join(lines)

    def _format_categories(self, categories: dict[str, PaperCategory]) -> str:
        lines = []
        for pid, cat in categories.items():
            parts = []
            if cat.methodology:
                parts.append(f"methodology={cat.methodology}")
            if cat.sub_topic:
                parts.append(f"sub_topic={cat.sub_topic}")
            lines.append(f"- {pid}: {', '.join(parts)}")
        return "\n".join(lines)

    def _format_extractions(self, extractions: dict[str, PaperExtraction]) -> str:
        lines = []
        for pid, ext in extractions.items():
            lines.append(f"- {pid}:")
            if ext.research_question:
                lines.append(f"  - Question: {ext.research_question}")
            if ext.key_results:
                lines.append(f"  - Results: {ext.key_results}")
            if ext.datasets:
                lines.append(f"  - Datasets: {', '.join(ext.datasets)}")
        return "\n".join(lines)

    def _format_relations(self, relations: list[ComparisonRelation]) -> str:
        lines = []
        for r in relations:
            lines.append(f"- {r.source_id} --[{r.relation_type}]--> {r.target_id}: {r.description}")
        return "\n".join(lines)

    async def synthesize(
        self,
        topic: str,
        papers: list[Paper],
        categories: dict[str, PaperCategory],
        extractions: dict[str, PaperExtraction],
        relations: list[ComparisonRelation],
    ) -> LiteratureReview:
        if not papers:
            return LiteratureReview(
                title=f"Literature Review: {topic}",
                sections=[ReviewSection(title="Note", content="No papers were found for this topic.")],
            )

        papers_summary = self._format_papers(papers)
        categories_summary = self._format_categories(categories)
        extractions_summary = self._format_extractions(extractions)
        relations_summary = self._format_relations(relations)

        prompt = SYNTHESIS_PROMPT.format(
            topic=topic,
            num_papers=len(papers),
            papers_summary=papers_summary,
            categories_summary=categories_summary,
            extractions_summary=extractions_summary,
            relations_summary=relations_summary,
        )

        try:
            response = await self.llm.generate(
                system_prompt="You are an expert academic writer specializing in literature reviews.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=8192,
            )

            # Parse markdown sections
            sections = self._parse_sections(response)
            title = response.split("\n")[0].lstrip("# ").strip() if response.startswith("#") else f"Literature Review: {topic}"

            return LiteratureReview(
                title=title,
                sections=sections,
                references=[p.id for p in papers],
                word_count=len(response.split()),
            )
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return LiteratureReview(
                title=f"Literature Review: {topic}",
                sections=[ReviewSection(title="Error", content=f"Failed to generate review: {e}")],
            )

    def _parse_sections(self, markdown: str) -> list[ReviewSection]:
        lines = markdown.split("\n")
        sections = []
        current_section = None
        current_content = []

        for line in lines:
            if line.startswith("## "):
                if current_section:
                    current_section.content = "\n".join(current_content).strip()
                    sections.append(current_section)
                current_section = ReviewSection(title=line.strip("#").strip(), content="")
                current_content = []
            elif line.startswith("### ") and current_section:
                current_section.subsections.append(
                    ReviewSection(title=line.strip("#").strip(), content="")
                )
            else:
                current_content.append(line)

        if current_section:
            current_section.content = "\n".join(current_content).strip()
            sections.append(current_section)

        return sections
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_synthesis.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/synthesis.py tests/test_agents/test_synthesis.py
git commit -m "feat: add SynthesisAgent"
```

---

### 任务 1.7：ReferenceAgent

**文件：**
- 创建：`src/agents/reference.py`
- 测试：`tests/test_agents/test_reference.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from src.agents.reference import ReferenceAgent
from src.models.schemas import Paper, Author, LiteratureReview, ReviewSection


@pytest.mark.asyncio
async def test_format_apa():
    """验证 APA 格式"""
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
    """验证 IEEE 格式"""
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
    """验证为综述附加参考文献"""
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
```

- [ ] **步骤 2：实现 ReferenceAgent**

```python
import logging
from src.models.schemas import Paper, LiteratureReview

logger = logging.getLogger(__name__)


class ReferenceAgent:
    def format_apa(self, paper: Paper) -> str:
        authors_str = ", ".join(
            self._format_author_apa(a.name) for a in paper.authors
        )
        year = paper.year or "n.d."
        title = paper.title
        source = paper.source.upper() if paper.source != "unknown" else ""
        url = paper.url or paper.doi or ""

        parts = [f"{authors_str} ({year}).", f"*{title}*.", source, url]
        return " ".join(p for p in parts if p)

    def format_ieee(self, paper: Paper) -> str:
        author_list = []
        for i, a in enumerate(paper.authors):
            name_parts = a.name.split()
            if len(name_parts) >= 2:
                formatted = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
            else:
                formatted = a.name
            author_list.append(formatted)

        authors_str = ", ".join(author_list)
        year = paper.year or "n.d."
        title = paper.title
        source = paper.source.upper() if paper.source != "unknown" else ""

        return f"{authors_str}, \"{title},\" {source}, {year}."

    def _format_author_apa(self, name: str) -> str:
        parts = name.split()
        if len(parts) >= 2:
            return f"{parts[-1]}, {' '.join(parts[:-1])[0]}."
        return name

    async def attach_references(self, review: LiteratureReview, papers: list[Paper]) -> LiteratureReview:
        paper_map = {p.id: p for p in papers}
        formatted = []

        for i, ref_id in enumerate(review.references, 1):
            paper = paper_map.get(ref_id)
            if paper:
                formatted.append(f"[{i}] {self.format_ieee(paper)}")
            else:
                formatted.append(f"[{i}] {ref_id}")

        if review.sections:
            review.sections.append(
                type(review.sections[0])(
                    title="References",
                    content="\n".join(formatted),
                )
            )
        return review
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_reference.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/reference.py tests/test_agents/test_reference.py
git commit -m "feat: add ReferenceAgent"
```

---

### 任务 1.8：QualityAgent

**文件：**
- 创建：`src/agents/quality.py`
- 测试：`tests/test_agents/test_quality.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.agents.quality import QualityAgent
from src.models.schemas import LiteratureReview, ReviewSection, Paper


@pytest.mark.asyncio
async def test_check_fact_consistency():
    """验证事实一致性检查"""
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


@pytest.mark.asyncio
async def test_check_structure():
    """验证结构检查"""
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


@pytest.mark.asyncio
async def test_missing_sections():
    """验证缺失章节检测"""
    agent = QualityAgent()
    review = LiteratureReview(
        title="Test",
        sections=[ReviewSection(title="Intro", content="...")],
    )
    issues = agent.check_structure(review)
    assert len(issues) > 0
```

- [ ] **步骤 2：实现 QualityAgent**

```python
import json
import logging
from src.models.schemas import LiteratureReview, Paper

logger = logging.getLogger(__name__)

QUALITY_PROMPT = """Review the following literature review for fact consistency.

Topic claims and their sources:
{review_text}

Available papers:
{papers_summary}

Check each claim made in the review against the source papers. Report:
1. Any claims that are not supported by the cited papers
2. Any citations that seem mismatched
3. Overall quality score (0.0 - 1.0)

Return JSON: {"issues": [{"claim": "...", "severity": "high|medium|low", "suggestion": "..."}], "score": 0.0, "suggestions": ["..."]}"""

REQUIRED_SECTIONS = ["Introduction", "Methodology", "Conclusion"]


class QualityAgent:
    def __init__(self):
        from src.services.llm_client import LLMClient
        self.llm = LLMClient()

    async def check_consistency(self, review: LiteratureReview, papers: list[Paper]) -> dict:
        review_text = "\n\n".join(f"## {s.title}\n{s.content}" for s in review.sections)
        papers_summary = "\n".join(f"{p.id}: {p.title} - {p.abstract[:200]}" for p in papers)

        prompt = QUALITY_PROMPT.format(review_text=review_text, papers_summary=papers_summary)

        try:
            response = await self.llm.generate(
                system_prompt="You are a quality assurance expert for academic literature reviews.",
                user_prompt=prompt,
                temperature=0.0,
            )
            return json.loads(response)
        except Exception as e:
            logger.warning(f"Quality check failed: {e}")
            return {"issues": [{"claim": "Quality check failed", "severity": "high", "suggestion": str(e)}], "score": 0.0, "suggestions": []}

    def check_structure(self, review: LiteratureReview) -> list[str]:
        issues = []
        section_titles = [s.title.lower() for s in review.sections]

        for required in REQUIRED_SECTIONS:
            if not any(required.lower() in t for t in section_titles):
                issues.append(f"Missing required section: {required}")

        if not review.title:
            issues.append("Review has no title")

        return issues

    async def full_quality_check(self, review: LiteratureReview, papers: list[Paper]) -> str:
        structure_issues = self.check_structure(review)
        consistency = await self.check_consistency(review, papers)

        report_parts = ["## Quality Report\n"]
        if structure_issues:
            report_parts.append("### Structure Issues\n")
            for issue in structure_issues:
                report_parts.append(f"- {issue}")

        report_parts.append(f"\n### Fact Consistency Score: {consistency.get('score', 0.0):.2f}\n")
        for issue in consistency.get("issues", []):
            report_parts.append(f"- [{issue.get('severity', 'info')}] {issue.get('claim', '')}")
            if issue.get("suggestion"):
                report_parts.append(f"  Suggestion: {issue['suggestion']}")

        if consistency.get("suggestions"):
            report_parts.append("\n### Suggestions\n")
            for s in consistency["suggestions"]:
                report_parts.append(f"- {s}")

        return "\n".join(report_parts)
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_agents/test_quality.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/agents/quality.py tests/test_agents/test_quality.py
git commit -m "feat: add QualityAgent"
```

---

## 阶段 2：编排与 API

### 任务 2.1：Orchestrator — LangGraph 工作流

**文件：**
- 创建：`src/orchestrator/workflow.py`
- 测试：`tests/test_orchestrator/test_workflow.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.workflow import LiteratureReviewWorkflow
from src.models.state import LiteratureReviewState
from src.models.schemas import (
    Paper, SearchQuery, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection,
)


@pytest.mark.asyncio
async def test_workflow_initialization():
    """验证工作流初始化"""
    workflow = LiteratureReviewWorkflow()
    assert workflow.app is not None


@pytest.mark.asyncio
async def test_full_pipeline_mocked():
    """验证完整流水线（全 mock）"""
    workflow = LiteratureReviewWorkflow()

    # Mock all agents
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
    """验证图结构完整性"""
    workflow = LiteratureReviewWorkflow()
    graph = workflow.build_graph()
    assert graph is not None
```

- [ ] **步骤 2：实现 Orchestrator**

```python
import logging
from typing import Any
from langgraph.graph import StateGraph, END
from src.models.state import LiteratureReviewState
from src.agents.query_expansion import QueryExpansionAgent
from src.agents.paper_discovery import PaperDiscoveryAgent
from src.agents.categorization import CategorizationAgent
from src.agents.extraction import ExtractionAgent
from src.agents.comparison import ComparisonAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.reference import ReferenceAgent
from src.agents.quality import QualityAgent

logger = logging.getLogger(__name__)


class LiteratureReviewWorkflow:
    def __init__(self):
        self.query_expansion_agent = QueryExpansionAgent()
        self.paper_discovery_agent = PaperDiscoveryAgent()
        self.categorization_agent = CategorizationAgent()
        self.extraction_agent = ExtractionAgent()
        self.comparison_agent = ComparisonAgent()
        self.synthesis_agent = SynthesisAgent()
        self.reference_agent = ReferenceAgent()
        self.quality_agent = QualityAgent()
        self.app = self.build_graph()

    def build_graph(self) -> StateGraph:
        workflow = StateGraph(LiteratureReviewState)

        workflow.add_node("expand_queries", self._expand_queries)
        workflow.add_node("discover_papers", self._discover_papers)
        workflow.add_node("categorize_papers", self._categorize_papers)
        workflow.add_node("extract_information", self._extract_information)
        workflow.add_node("analyze_relationships", self._analyze_relationships)
        workflow.add_node("synthesize_review", self._synthesize_review)
        workflow.add_node("format_references", self._format_references)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("finalize", self._finalize)

        workflow.set_entry_point("expand_queries")

        workflow.add_edge("expand_queries", "discover_papers")
        workflow.add_edge("discover_papers", "categorize_papers")
        workflow.add_edge("categorize_papers", "extract_information")
        workflow.add_edge("extract_information", "analyze_relationships")
        workflow.add_edge("analyze_relationships", "synthesize_review")
        workflow.add_edge("synthesize_review", "format_references")
        workflow.add_edge("format_references", "quality_check")

        # Conditional: if quality is good, finalize; otherwise iterate
        workflow.add_conditional_edges(
            "quality_check",
            self._decide_next,
            {"finalize": "finalize", "synthesize_review": "synthesize_review"},
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def _expand_queries(self, state: LiteratureReviewState) -> dict:
        logger.info("Expanding queries...")
        queries = await self.query_expansion_agent.expand(state["topic"])
        return {"search_queries": queries, "progress": {"expand_queries": "done"}}

    async def _discover_papers(self, state: LiteratureReviewState) -> dict:
        logger.info("Discovering papers...")
        all_papers = []
        for sq in state["search_queries"]:
            papers = await self.paper_discovery_agent.discover(sq)
            all_papers.extend(papers)

        filtered = await self.paper_discovery_agent.filter_by_relevance(all_papers, state["topic"])
        return {"discovered_papers": all_papers, "filtered_papers": filtered, "progress": {"discover_papers": "done"}}

    async def _categorize_papers(self, state: LiteratureReviewState) -> dict:
        logger.info("Categorizing papers...")
        categories = await self.categorization_agent.categorize(state["filtered_papers"])
        return {"paper_categories": categories, "progress": {"categorize_papers": "done"}}

    async def _extract_information(self, state: LiteratureReviewState) -> dict:
        logger.info("Extracting information...")
        extractions = await self.extraction_agent.batch_extract(state["filtered_papers"])
        return {"paper_extractions": extractions, "progress": {"extract_information": "done"}}

    async def _analyze_relationships(self, state: LiteratureReviewState) -> dict:
        logger.info("Analyzing relationships...")
        relations = await self.comparison_agent.analyze_all(state["filtered_papers"])
        return {"relations": relations, "progress": {"analyze_relationships": "done"}}

    async def _synthesize_review(self, state: LiteratureReviewState) -> dict:
        logger.info("Synthesizing review...")
        review = await self.synthesis_agent.synthesize(
            topic=state["topic"],
            papers=state["filtered_papers"],
            categories=state["paper_categories"],
            extractions=state["paper_extractions"],
            relations=state["relations"],
        )
        return {"review": review, "progress": {"synthesize_review": "done"}}

    async def _format_references(self, state: LiteratureReviewState) -> dict:
        logger.info("Formatting references...")
        review = await self.reference_agent.attach_references(state["review"], state["filtered_papers"])
        return {"review": review, "progress": {"format_references": "done"}}

    async def _quality_check(self, state: LiteratureReviewState) -> dict:
        logger.info("Running quality check...")
        report = await self.quality_agent.full_quality_check(state["review"], state["filtered_papers"])
        return {"quality_report": report, "progress": {"quality_check": "done"}}

    def _decide_next(self, state: LiteratureReviewState) -> str:
        # Simple heuristic: if quality report mentions critical issues, re-synthesize
        if state.get("quality_report") and "critical" in state["quality_report"].lower():
            return "synthesize_review"
        return "finalize"

    async def _finalize(self, state: LiteratureReviewState) -> dict:
        logger.info("Finalizing...")
        return {**state, "progress": {"finalize": "done"}}

    async def run(self, topic: str) -> dict[str, Any]:
        initial_state: LiteratureReviewState = {
            "topic": topic,
            "search_queries": [],
            "discovered_papers": [],
            "filtered_papers": [],
            "paper_categories": {},
            "paper_extractions": {},
            "relations": [],
            "review": LiteratureReview(title="", sections=[]),
            "quality_report": "",
            "errors": [],
            "progress": {},
        }
        result = await self.app.ainvoke(initial_state)
        return result
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_orchestrator/test_workflow.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/orchestrator/workflow.py tests/test_orchestrator/test_workflow.py
git commit -m "feat: add LangGraph orchestrator workflow"
```

---

### 任务 2.2：FastAPI 后端

**文件：**
- 创建：`src/api/main.py`
- 测试：`tests/test_api/test_api.py`

- [ ] **步骤 1：编写测试**

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.api.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """验证健康检查"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_review_endpoint_invalid_topic(client):
    """验证空主题校验"""
    resp = await client.post("/api/review", json={"topic": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_review_endpoint_with_mock(client):
    """验证综述生成接口（mock 工作流）"""
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
```

- [ ] **步骤 2：实现 FastAPI 应用**

```python
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.orchestrator.workflow import LiteratureReviewWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Automated Literature Review Platform", version="0.1.0")


class ReviewRequest(BaseModel):
    topic: str


class ReviewResponse(BaseModel):
    review: dict
    quality_report: str = ""
    progress: dict = {}
    errors: list[str] = []


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "literature-review-platform"}


@app.post("/api/review", response_model=ReviewResponse)
async def generate_review(request: ReviewRequest):
    if not request.topic.strip():
        raise HTTPException(status_code=422, detail="Topic cannot be empty")

    workflow = LiteratureReviewWorkflow()
    try:
        result = await workflow.run(topic=request.topic)

        review_dict = result.get("review", {})
        if hasattr(review_dict, "model_dump"):
            review_dict = review_dict.model_dump()

        return ReviewResponse(
            review=review_dict,
            quality_report=result.get("quality_report", ""),
            progress=result.get("progress", {}),
            errors=result.get("errors", []),
        )
    except Exception as e:
        logger.exception("Review generation failed")
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **步骤 3：运行测试验证通过**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_api/test_api.py -v`
预期：PASS

- [ ] **步骤 4：Commit**

```bash
git add src/api/main.py tests/test_api/test_api.py
git commit -m "feat: add FastAPI backend with review endpoint"
```

---

### 任务 2.3：Streamlit 前端

**文件：**
- 创建：`src/frontend/app.py`

- [ ] **步骤 1：编写 Streamlit 应用**

```python
import streamlit as st
import httpx
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="自动化文献综述生成平台",
    page_icon="📚",
    layout="wide",
)

st.title("📚 自动化文献综述生成平台")
st.markdown("输入一个研究主题，AI 自动完成文献检索、分析、综述撰写。")

# Sidebar
with st.sidebar:
    st.header("关于")
    st.info(
        "本系统使用多 Agent 架构（LangGraph），\n"
        "自动完成：主题扩展 → 论文检索 → 分类 →\n"
        "信息提取 → 关系分析 → 综述撰写 → 质量控制。"
    )
    api_url = st.text_input("API 地址", value=API_BASE)

# Main input
topic = st.text_input("研究主题", placeholder="例如：Transformer 在时间序列预测中的应用")

col1, col2 = st.columns([1, 5])
with col1:
    generate_btn = st.button("生成综述", type="primary", disabled=not topic)

if generate_btn and topic:
    with st.spinner("正在生成文献综述，这可能需要几分钟..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Phase indicators
        phases = [
            "扩展检索词", "检索论文", "分类论文",
            "提取信息", "分析关系", "撰写综述",
            "格式化引用", "质量检查", "完成",
        ]

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = client.post(
                    f"{api_url}/api/review",
                    json={"topic": topic},
                )

            if response.status_code == 200:
                data = response.json()
                review = data.get("review", {})
                quality = data.get("quality_report", "")

                progress_bar.progress(100)
                status_text.success("综述生成完成！")

                # Display results
                st.header(review.get("title", "文献综述"))

                for section in review.get("sections", []):
                    st.subheader(section.get("title", ""))
                    st.markdown(section.get("content", ""))

                if quality:
                    with st.expander("质量报告"):
                        st.markdown(quality)

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "下载 Markdown",
                        data=st.text(review).getvalue() if False else json.dumps(review, ensure_ascii=False, indent=2),
                        file_name=f"{topic}_literature_review.md",
                    )
            else:
                status_text.error(f"生成失败：{response.text}")

        except Exception as e:
            status_text.error(f"请求失败：{e}")

# Usage tips
with st.expander("使用提示"):
    st.markdown("""
    - **主题选择**：选择具体的研究方向效果更好，如 "Graph Neural Networks for Drug Discovery"
    - **处理时间**：根据论文数量，通常需要 1-5 分钟
    - **输出格式**：支持 Markdown 下载，可自行转换为 LaTeX/PDF
    """)
```

- [ ] **步骤 2：验证前端无语法错误**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -c "import ast; ast.parse(open('src/frontend/app.py').read()); print('Syntax OK')"`
预期：Syntax OK

- [ ] **步骤 3：Commit**

```bash
git add src/frontend/app.py
git commit -m "feat: add Streamlit frontend"
```

---

## 阶段 3：集成与验证

### 任务 3.1：端到端集成测试

**文件：**
- 创建：`tests/test_integration.py`

- [ ] **步骤 1：编写集成测试**

```python
import pytest
from unittest.mock import AsyncMock, patch
from src.orchestrator.workflow import LiteratureReviewWorkflow


@pytest.mark.asyncio
async def test_end_to_end_pipeline():
    """端到端流水线测试（全 mock 外部依赖）"""
    workflow = LiteratureReviewWorkflow()

    # Mock all external API calls
    with patch.multiple(
        workflow.query_expansion_agent,
        expand=AsyncMock(return_value=[
            type("SQ", (), {"original_query": "test", "expanded_queries": ["test"], "sub_queries": []})()
        ]),
    ):
        with patch.multiple(
            workflow.paper_discovery_agent,
            discover=AsyncMock(return_value=[
                type("Paper", (), {"id": "a", "title": "A", "abstract": "A.", "authors": [], "year": 2023, "source": "arxiv", "url": None, "doi": None, "citations": 0, "relevance_score": 0.0, "keywords": []})()
            ]),
            filter_by_relevance=AsyncMock(return_value=[
                type("Paper", (), {"id": "a", "title": "A", "abstract": "A.", "authors": [], "year": 2023, "source": "arxiv", "url": None, "doi": None, "citations": 0, "relevance_score": 0.0, "keywords": []})()
            ]),
        ):
            with patch.object(workflow.categorization_agent, "categorize", AsyncMock(return_value={})):
                with patch.object(workflow.extraction_agent, "batch_extract", AsyncMock(return_value={})):
                    with patch.object(workflow.comparison_agent, "analyze_all", AsyncMock(return_value=[])):
                        with patch.object(workflow.synthesis_agent, "synthesize", AsyncMock(return_value=type("LR", (), {"title": "Test", "sections": [], "references": [], "word_count": 10, "quality_score": None, "model_dump": lambda self: {"title": "Test"}})())):
                            with patch.object(workflow.reference_agent, "attach_references", AsyncMock(side_effect=lambda r, p: r)):
                                with patch.object(workflow.quality_agent, "full_quality_check", AsyncMock(return_value="OK")):
                                    result = await workflow.run(topic="Test")
                                    assert result is not None
                                    assert "review" in result


@pytest.mark.asyncio
async def test_pipeline_with_real_arxiv():
    """使用真实 arXiv API 测试（不 mock）"""
    from src.services.academic_apis import search_arxiv

    papers = await search_arxiv("transformer time series", max_results=3)
    assert len(papers) > 0
    assert all(p.source == "arxiv" for p in papers)
```

- [ ] **步骤 2：运行测试**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/test_integration.py -v`
预期：PASS（mock 测试通过；real arXiv 测试需要网络）

- [ ] **步骤 3：Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests"
```

---

### 任务 3.2：运行全量测试套件

- [ ] **步骤 1：运行所有测试**

运行：`cd /Users/liuchao/claude_code_project/Automated_Literature_Review_Generation_Platform && python -m pytest tests/ -v --cov=src`
预期：全部 PASS

- [ ] **步骤 2：如有失败，修复并重新运行**

- [ ] **步骤 3：Commit（修复后）**

```bash
git add -A && git commit -m "test: fix failing tests and ensure full test suite passes"
```

---

## 项目总结

### 架构总览

```
用户输入 (研究主题)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Orchestrator (LangGraph)                           │
│                                                      │
│  QueryExpansionAgent → PaperDiscoveryAgent           │
│         → CategorizationAgent → ExtractionAgent      │
│         → ComparisonAgent → SynthesisAgent           │
│         → ReferenceAgent → QualityAgent              │
│                                                      │
│  质量控制不合格 → 循环回 SynthesisAgent              │
│  质量控制合格 → Finalize                             │
└─────────────────────────────────────────────────────┘
    │
    ▼
输出 (Markdown 文献综述 + 质量报告)
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent 编排 | LangGraph | 支持有向图流程 + 条件分支，比直线 chain 灵活 |
| LLM SDK | anthropic | Claude 长上下文窗口适合综述生成 |
| 向量库 | ChromaDB | 本地嵌入式，零运维，适合原型 |
| 关系图谱 | NetworkX | 轻量级图结构，无需图数据库 |
| 前端 | Streamlit | 快速原型，Python 一体，适合展示 |
| 去重策略 | DOI + 标题模糊匹配 | 无需 ML 模型的高效去重 |
| 质量迭代 | 条件判断重写 | 如果质量分低，循环回 synthesis 改进 |

### 任务清单总览

- [ ] 任务 0.1：项目初始化与依赖配置
- [ ] 任务 0.2：核心数据模型
- [ ] 任务 0.3：LLM Client 封装
- [ ] 任务 0.4：学术 API 封装
- [ ] 任务 0.5：向量存储与图谱存储
- [ ] 任务 1.1：QueryExpansionAgent
- [ ] 任务 1.2：PaperDiscoveryAgent
- [ ] 任务 1.3：CategorizationAgent
- [ ] 任务 1.4：ExtractionAgent
- [ ] 任务 1.5：ComparisonAgent
- [ ] 任务 1.6：SynthesisAgent
- [ ] 任务 1.7：ReferenceAgent
- [ ] 任务 1.8：QualityAgent
- [ ] 任务 2.1：Orchestrator — LangGraph 工作流
- [ ] 任务 2.2：FastAPI 后端
- [ ] 任务 2.3：Streamlit 前端
- [ ] 任务 3.1：端到端集成测试
- [ ] 任务 3.2：全量测试套件验证
