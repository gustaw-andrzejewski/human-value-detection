import time
from collections import defaultdict
from functools import partial
from multiprocessing import Manager, Pool
from typing import Dict, List, Tuple

import numpy as np
from datasets import DatasetDict, concatenate_datasets, load_dataset
from py2neo import Graph, Node, Relationship
from tqdm import tqdm

from neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

ARGUMENT = "ARGUMENT"
ARGUMENT_ID = "Argument ID"
CONCLUSION = "Conclusion"
LABELS = "Labels"
PREMISE = "Premise"
SHARES_VALUES_WITH = "SHARES_VALUES_WITH"
STANCE = "Stance"

SIMILARITY_THRESHOLD = 0.5


def load_data() -> Tuple[DatasetDict, dict]:
    """Loads the dataset and returns the concatenated version."""
    eval_dataset = load_dataset("webis/Touche23-ValueEval", revision="main")
    dataset = concatenate_datasets(
        [eval_dataset["train"], eval_dataset["validation"], eval_dataset["test"]]
    )
    argument_id_map = {item[ARGUMENT_ID]: idx for idx, item in enumerate(dataset)}
    return dataset, argument_id_map


def create_inverted_index(dataset: DatasetDict) -> Dict:
    """
    Creates an inverted index for the dataset.
    """
    inverted_index = defaultdict(list)
    for i, example in tqdm(enumerate(dataset), "Creating inverted index"):
        labels = example[LABELS]
        argument_id = example[ARGUMENT_ID]
        for label_pos, label in enumerate(labels):
            if label == 1:
                inverted_index[label_pos].append(argument_id)
    return inverted_index


def compute_pairwise_similarity(
    dataset: DatasetDict,
    argument_id_map: Dict[str, int],
    base_id: str,
    comparison_ids: List[str],
    similarity_dict: Dict,
) -> None:
    """
    Calculates pairwise similarity for the dataset.
    """
    for id2 in [id for id in comparison_ids if id != base_id]:
        indices_pair = tuple(sorted((base_id, id2)))

        if indices_pair not in similarity_dict:
            labels1 = dataset[argument_id_map[base_id]][LABELS]
            labels2 = dataset[argument_id_map[id2]][LABELS]
            intersection = np.sum(np.logical_and(labels1, labels2))
            union = np.sum(np.logical_or(labels1, labels2))
            similarity_dict[indices_pair] = intersection / union if union != 0 else 0


def calculate_similarity_matrix(
    inverted_index: Dict, dataset: DatasetDict, argument_id_map: Dict[str, int]
) -> Dict:
    """
    Calculates a similarity matrix for the given data.
    """
    with Manager() as manager:
        start_time = time.time()
        similarity_dict = manager.dict()

        with Pool() as pool:
            for label, ids in tqdm(
                inverted_index.items(), "Calculating similarity between arguments"
            ):
                func = partial(
                    compute_pairwise_similarity,
                    dataset,
                    argument_id_map,
                    comparison_ids=ids,
                    similarity_dict=similarity_dict,
                )
                pool.map(func, ids)

        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

        return dict(similarity_dict)


def create_neo4j_database(dataset: DatasetDict, similarity_dict: Dict) -> None:
    """
    Creates a graph with nodes representing arguments and edges representing similarity between arguments.
    """
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    for sample in tqdm(dataset, "Creating nodes"):
        node = Node(
            ARGUMENT,
            id=sample[ARGUMENT_ID],
            stance=sample[STANCE],
            premise=sample[PREMISE],
            conclusion=sample[CONCLUSION],
            labels=sample[LABELS],
        )
        graph.create(node)

    with tqdm(similarity_dict.items(), "Creating edges") as t:
        for ((id1, id2), similarity) in t:
            if similarity >= SIMILARITY_THRESHOLD:
                node1 = graph.nodes.match(ARGUMENT, id=id1).first()
                node2 = graph.nodes.match(ARGUMENT, id=id2).first()
                rel = Relationship(
                    node1, SHARES_VALUES_WITH, node2, similarity=float(similarity)
                )
                graph.create(rel)
                rel = Relationship(
                    node2, SHARES_VALUES_WITH, node1, similarity=float(similarity)
                )
                graph.create(rel)

                t.set_description(f"Creating edges - added: {t.n} edges")
                t.refresh()


if __name__ == "__main__":
    dataset, argument_id_map = load_data()
    inverted_index = create_inverted_index(dataset)
    similarity_dict = calculate_similarity_matrix(
        inverted_index, dataset, argument_id_map
    )
    create_neo4j_database(dataset, similarity_dict)
