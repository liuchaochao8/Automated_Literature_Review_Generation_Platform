import logging
from src.models.schemas import (
    Paper, PaperCategory, PaperExtraction,
    ComparisonRelation, LiteratureReview, ReviewSection,
)

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
        current_subsection = None
        section_content = []  # 只存放第一个 ### 之前的内容
        sub_content = []
        seen_subsection = False

        for line in lines:
            if line.startswith("## "):
                # save previous section + subsection
                if current_section:
                    if current_subsection:
                        current_subsection.content = "\n".join(sub_content).strip()
                    current_section.content = "\n".join(section_content).strip()
                    sections.append(current_section)
                current_section = ReviewSection(title=line.strip("#").strip(), content="")
                current_subsection = None
                section_content = []
                sub_content = []
                seen_subsection = False
            elif line.startswith("### ") and current_section:
                # save previous subsection content
                if current_subsection:
                    current_subsection.content = "\n".join(sub_content).strip()
                current_subsection = ReviewSection(title=line.strip("#").strip(), content="")
                current_section.subsections.append(current_subsection)
                sub_content = []
                seen_subsection = True
            else:
                if not seen_subsection:
                    section_content.append(line)
                else:
                    sub_content.append(line)

        if current_section:
            if current_subsection:
                current_subsection.content = "\n".join(sub_content).strip()
            current_section.content = "\n".join(section_content).strip()
            sections.append(current_section)

        return sections
