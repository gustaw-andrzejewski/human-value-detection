from pathlib import Path

import numpy as np
from py2neo import Graph

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

LABELS_FILENAME = Path("node_labels.npy")
DATA_FOLDER = Path("workspace/data/raw")


def load_labels_from_neo4j(
    database_uri: str, database_username: str, database_password: str
) -> list:
    """
    Loads the labels from the Neo4j database.
    """
    graph = Graph(database_uri, auth=(database_username, database_password))
    query = """
    MATCH (a:ARGUMENT)
    RETURN a.id, a.labels
    ORDER BY a.id
    """
    results = graph.run(query)
    return np.array([result["a.labels"] for result in results])


def save_labels(labels: np.ndarray, output_path: Path) -> None:
    """
    Saves the labels to the given path.
    """
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
    np.save(DATA_FOLDER / output_path, labels)


if __name__ == "__main__":
    labels = load_labels_from_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    save_labels(labels, LABELS_FILENAME)
