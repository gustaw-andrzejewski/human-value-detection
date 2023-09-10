from pathlib import Path
from typing import Tuple

import numpy as np
from py2neo import Graph

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

ARG_EDGE_INDEX_FILENAME = Path("arg_edge_index.npy")
ARG_EDGE_ATTR_FILENAME = Path("arg_edge_attributes.npy")
DATA_FOLDER = Path("workspace/data/raw")


def get_neo4j_db(database_uri: str, database_username: str, database_password: str) -> Graph:
    return Graph(database_uri, auth=(database_username, database_password))


def get_unique_node_ids(database_connection: Graph) -> list:
    unique_ids_query = """
        MATCH (n)
        WHERE n:ARGUMENT OR n:LABEL
        RETURN n.id AS id
        ORDER BY n.id
        """
    return database_connection.run(unique_ids_query).data()


def get_unique_relationships(database_connection: Graph, relationship_type: str) -> list:
    unique_relationships_query = f"""
        MATCH (n1)-[r:{relationship_type}]-(n2)
        WHERE n1.id < n2.id
        RETURN n1.id AS source, n2.id AS target, r.weight AS weight
        ORDER BY n1.id, n2.id
        """
    return database_connection.run(unique_relationships_query).data()


def create_edge_index_and_attributes(
    unique_relationships: list, unique_nodes: list
) -> Tuple[np.ndarray, np.ndarray]:
    node_to_idx = {node_id: idx for idx, node_id in enumerate([item["id"] for item in unique_nodes])}
    edge_pairs = [(node_to_idx[item["source"]], node_to_idx[item["target"]]) for item in unique_relationships]
    edge_attr = [item["weight"] for item in unique_relationships]

    edge_index = np.array([[src, tgt] for src, tgt in edge_pairs] + [[tgt, src] for src, tgt in edge_pairs])
    edge_attr = np.array(edge_attr + edge_attr)
    return edge_index, edge_attr


def save_edge_index_and_attributes(
    edge_index: np.ndarray, edge_attr: np.ndarray, edge_index_filename, edge_attr_filename
) -> None:
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
    np.save(DATA_FOLDER / edge_index_filename, edge_index.transpose())
    np.save(DATA_FOLDER / edge_attr_filename, edge_attr)


if __name__ == "__main__":
    neo4j_db = get_neo4j_db(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    unique_node_ids = get_unique_node_ids(neo4j_db)

    arg_relationships = get_unique_relationships(neo4j_db, "SIMILAR_TO")
    arg_edge_index, arg_edge_attr = create_edge_index_and_attributes(arg_relationships, unique_node_ids)
    save_edge_index_and_attributes(
        arg_edge_index, arg_edge_attr, ARG_EDGE_INDEX_FILENAME, ARG_EDGE_ATTR_FILENAME
    )