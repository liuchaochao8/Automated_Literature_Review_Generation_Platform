import networkx as nx
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

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def get_aggregated_view(self) -> dict:
        return {
            "total_papers": self.node_count(),
            "total_relations": self.edge_count(),
            "density": round(nx.density(self.graph), 4),
        }
