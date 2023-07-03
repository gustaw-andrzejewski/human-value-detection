from pathlib import Path
from typing import Tuple

import numpy as np
from py2neo import Graph

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

EDGE_INDEX_FILENAME = Path("edge_index.npy")
EDGE_ATTR_FILENAME = Path("edge_attributes.npy")
DATA_FOLDER = Path("workspace/data/")


def get_neo4j_db(
    database_uri: str, database_username: str, database_password: str
) -> Graph:
    """
    Connects to the neo4j database
    """
    return Graph(database_uri, auth=(database_username, database_password))


def get_unique_node_ids(database_connection: Graph) -> list:
    """
    Fetches unique node ids
    """
    unique_ids_query = """
        MATCH (a:ARGUMENT)
        RETURN a.id AS id
        ORDER BY a.id
        """
    return database_connection.run(unique_ids_query).data()


def get_unique_relationships(database_connection: Graph) -> list:
    """
    Fetches unique relationships between nodes
    """
    unique_relationships_query = """
        MATCH (a1:ARGUMENT)-[r:SHARES_VALUES_WITH]-(a2:ARGUMENT)
        WHERE a1.id < a2.id
        RETURN a1.id AS source, a2.id AS target, r.similarity AS similarity
        ORDER BY a1.id, a2.id
        """
    return database_connection.run(unique_relationships_query).data()


def create_edge_index_and_attributes(
    unique_relationships: list, unique_nodes: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates edge index and attributes numpy arrays to get data
    for training a graph neural network
    """
    node_to_idx = {
        node_id: idx
        for idx, node_id in enumerate([item["id"] for item in unique_nodes])
    }
    edge_pairs = [
        (node_to_idx[item["source"]], node_to_idx[item["target"]])
        for item in unique_relationships
    ]
    edge_attr = [item["similarity"] for item in unique_relationships]

    edge_index = np.array(
        [[src, tgt] for src, tgt in edge_pairs]
        + [[tgt, src] for src, tgt in edge_pairs]
    )
    edge_attr = np.array(edge_attr + edge_attr)
    return edge_index, edge_attr


def save_edge_index_and_attributes(
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    edge_index_filename,
    edge_attr_filename,
) -> None:
    """
    Saves edge index and attributes to numpy files
    """
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
    np.save(DATA_FOLDER / edge_index_filename, edge_index)
    np.save(DATA_FOLDER / edge_attr_filename, edge_attr)


if __name__ == "__main__":
    neo4j_db = get_neo4j_db(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    unique_node_ids = get_unique_node_ids(neo4j_db)
    print(f"Fetched {len(unique_node_ids)} unique nodes.")
    unique_relationships = get_unique_relationships(neo4j_db)
    print(f"Fetched {len(unique_relationships)} unique relationships.")
    edge_index, edge_attr = create_edge_index_and_attributes(
        unique_relationships, unique_node_ids
    )
    print("Edge index and attributes created.")
    save_edge_index_and_attributes(
        edge_index, edge_attr, EDGE_INDEX_FILENAME, EDGE_ATTR_FILENAME
    )
    print(f"Edge index and attributes saved to {DATA_FOLDER}.")
