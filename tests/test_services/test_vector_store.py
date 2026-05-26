import pytest
import tempfile
from src.services.vector_store import PaperVectorStore
from src.models.schemas import Paper


@pytest.fixture
def vector_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PaperVectorStore(persist_dir=tmpdir)
        yield store


def test_add_and_count_papers(vector_store):
    paper = Paper(
        id="test:001",
        title="Machine Learning Basics",
        abstract="Introduction to machine learning concepts and algorithms.",
    )
    vector_store.add_paper(paper, embedding=[0.1] * 128)
    assert vector_store.count() == 1


def test_search_similar(vector_store):
    papers = [
        Paper(id="test:001", title="Deep Learning", abstract="Neural networks."),
        Paper(id="test:002", title="Machine Learning", abstract="Statistical methods."),
    ]
    for p in papers:
        vector_store.add_paper(p, embedding=[0.1] * 128)

    results = vector_store.search([0.1] * 128, top_k=2)
    assert len(results) == 2


def test_delete_paper(vector_store):
    paper = Paper(id="test:001", title="Test", abstract="Test abstract.")
    vector_store.add_paper(paper, embedding=[0.1] * 128)
    vector_store.delete_paper("test:001")
    assert vector_store.count() == 0
