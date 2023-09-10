from pathlib import Path

import numpy as np
from py2neo import Graph, Relationship
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

DATA_FOLDER = Path("workspace/data/raw")
EMBEDDINGS_FILENME = Path("arguments_bert_embeddings.npy")
LABEL_EMBEDDINGS_FILENAME = Path("labels_bert_embeddings.npy")
ARGUMENT = "ARGUMENT"
ARGUMENT_ID = "Argument ID"
CONCLUSION = "Conclusion"
LABELS = "Labels"
PREMISE = "Premise"
STANCE = "Stance"
LABEL = "LABEL"


def load_embeddings(file_path: Path) -> np.ndarray:
    """Loads the saved embeddings from the given path."""
    return np.load(DATA_FOLDER / file_path)


def load_arguments_from_neo4j(
    database_uri: str, database_username: str, database_password: str
) -> list:
    """Loads the arguments from the Neo4j database."""
    print("Loading arguments from Neo4j")
    graph = Graph(database_uri, auth=(database_username, database_password))
    query = """
    MATCH (a:ARGUMENT)
    RETURN a.id AS id, a.Stance AS Stance
    ORDER BY a.id
    """
    arguments = graph.run(query).data()
    print("Finished loading arguments from Neo4j")
    return arguments


def get_top_k_similar_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    """Returns the indices of the top k similar items for each item in the matrix."""
    sorted_indices = np.argsort(matrix, axis=1)[:, -k-1:-1]
    return np.fliplr(sorted_indices)


def create_similarity_edges(graph: Graph, arguments: list, embeddings: np.ndarray, k: int = 10):
    """
    Creates edges in the Neo4j graph based on similarity of embeddings.
    """
    print("Creating similarity edges")
    similarity_matrix = cosine_similarity(embeddings)
    top_k_indices = get_top_k_similar_indices(similarity_matrix, k)
    print("Finished calculating similarity matrix")

    for idx, similar_indices in tqdm(enumerate(top_k_indices), desc="Creating edges"):
        argument_node = graph.nodes.match(ARGUMENT, id=arguments[idx]['id']).first()
        for similar_idx in similar_indices:
            if arguments[idx][STANCE] == arguments[similar_idx][STANCE]:
                similar_node = graph.nodes.match(ARGUMENT, id=arguments[similar_idx]['id']).first()
                rel = Relationship(argument_node, "SIMILAR_TO", similar_node)
                rel["weight"] = float(similarity_matrix[idx, similar_idx])
                graph.merge(rel)
    print("Finished creating similarity edges")


def load_labels_from_neo4j(database_uri: str, database_username: str, database_password: str) -> list:
    """Loads the labels from the Neo4j database."""
    print("Loading labels from Neo4j")
    graph = Graph(database_uri, auth=(database_username, database_password))
    query = """
    MATCH (l:LABEL)
    RETURN l.id AS id, l.name AS name
    ORDER BY l.id
    """
    labels = graph.run(query).data()
    print("Finished loading labels from Neo4j")
    return labels


def create_label_edges_with_similarity(graph: Graph, labels: list, label_embeddings: np.ndarray):
    """
    Creates edges in the Neo4j graph connecting each label to every other label.
    The weight of the edge is the cosine similarity between the embeddings of the labels.
    """
    print("Creating label edges with similarity")

    similarity_matrix = cosine_similarity(label_embeddings)

    for idx, label_data in tqdm(enumerate(labels), desc="Creating label edges with similarity"):
        label_node = graph.nodes.match(LABEL, id=label_data['id']).first()
        for other_idx, other_label_data in enumerate(labels):
            if label_data['id'] != other_label_data['id']:
                other_label_node = graph.nodes.match(LABEL, id=other_label_data['id']).first()
                rel = Relationship(label_node, "RELATED_TO", other_label_node)
                rel["weight"] = float(similarity_matrix[idx, other_idx])
                graph.merge(rel)

    print("Finished creating label edges with similarity")


if __name__ == "__main__":
    argument_embeddings = load_embeddings(EMBEDDINGS_FILENME)
    label_embeddings = load_embeddings(LABEL_EMBEDDINGS_FILENAME)
    arguments = load_arguments_from_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    labels = load_labels_from_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    create_similarity_edges(graph, arguments, argument_embeddings)
    create_label_edges_with_similarity(graph, labels, label_embeddings)
