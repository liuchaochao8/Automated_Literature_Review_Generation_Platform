import json
import logging
from src.models.schemas import LiteratureReview, Paper
from src.services.llm_client import LLMClient

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

Return JSON: {{"issues": [{{"claim": "...", "severity": "high|medium|low", "suggestion": "..."}}], "score": 0.0, "suggestions": ["..."]}}"""

REQUIRED_SECTIONS = ["Introduction", "Methodology", "Conclusion"]


class QualityAgent:
    def __init__(self):
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
