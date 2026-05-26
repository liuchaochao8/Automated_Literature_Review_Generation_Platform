import pytest
from src.services.graph_store import CitationGraph
from src.models.schemas import ComparisonRelation


@pytest.fixture
def graph():
    return CitationGraph()


def test_add_relation(graph):
    rel = ComparisonRelation(
        source_id="paper:a", target_id="paper:b",
        relation_type="citation", description="A cites B", confidence=1.0,
    )
    graph.add_relation(rel)
    assert graph.node_count() == 2
    assert graph.edge_count() == 1


def test_get_related_papers(graph):
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
    graph.add_relation(ComparisonRelation(
        source_id="a", target_id="b", relation_type="citation", description="", confidence=1.0,
    ))
    graph.add_relation(ComparisonRelation(
        source_id="b", target_id="c", relation_type="citation", description="", confidence=1.0,
    ))
    path = graph.find_path("a", "c")
    assert len(path) == 3


def test_get_aggregated_view(graph):
    graph.add_relation(ComparisonRelation(
        source_id="a", target_id="b", relation_type="citation", description="", confidence=1.0,
    ))
    view = graph.get_aggregated_view()
    assert view["total_papers"] == 2
    assert view["total_relations"] == 1
