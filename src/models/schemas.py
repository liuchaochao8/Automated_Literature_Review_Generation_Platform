from pydantic import BaseModel, Field
from typing import Optional, Literal


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class Paper(BaseModel):
    id: str
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    year: Optional[int] = None
    source: str = "unknown"
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
    research_paradigm: Optional[str] = None
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
