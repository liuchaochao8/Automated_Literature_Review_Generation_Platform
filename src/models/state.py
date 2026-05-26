from typing import TypedDict
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
    paper_categories: dict[str, PaperCategory]

    # Phase 4: Extraction
    paper_extractions: dict[str, PaperExtraction]

    # Phase 5: Comparison
    relations: list[ComparisonRelation]

    # Phase 6: Synthesis
    review: LiteratureReview

    # Phase 7: Quality
    quality_report: str

    # Control
    errors: list[str]
    progress: dict[str, str]
