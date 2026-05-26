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
