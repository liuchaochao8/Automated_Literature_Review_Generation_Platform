import logging
from src.models.schemas import Paper, LiteratureReview

logger = logging.getLogger(__name__)


class ReferenceAgent:
    def format_apa(self, paper: Paper) -> str:
        authors_str = ", ".join(self._format_author_apa(a.name) for a in paper.authors)
        year = paper.year or "n.d."
        title = paper.title
        source = paper.source.upper() if paper.source != "unknown" else ""
        url = paper.url or paper.doi or ""

        parts = [f"{authors_str} ({year}).", f"*{title}*.", source, url]
        return " ".join(p for p in parts if p)

    def format_ieee(self, paper: Paper) -> str:
        year = paper.year or "n.d."
        title = paper.title
        source = paper.source.upper() if paper.source != "unknown" else ""

        if paper.authors:
            author_list = []
            for a in paper.authors:
                name_parts = a.name.split()
                if len(name_parts) >= 2:
                    formatted = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}"
                else:
                    formatted = a.name
                author_list.append(formatted)
            authors_str = ", ".join(author_list)
            return f"{authors_str}, \"{title},\" {source}, {year}."
        else:
            return f"\"{title},\" {source}, {year}."

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
