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
                "abstract": paper.abstract[:1000] if paper.abstract else "",
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
