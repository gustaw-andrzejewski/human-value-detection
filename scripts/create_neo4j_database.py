from datasets import DatasetDict, concatenate_datasets, load_dataset
from py2neo import Graph, Node
from tqdm import tqdm

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

ARGUMENT = "ARGUMENT"
ARGUMENT_ID = "Argument ID"
CONCLUSION = "Conclusion"
LABELS = "Labels"
PREMISE = "Premise"
STANCE = "Stance"

labels = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity",
]


def load_data() -> DatasetDict:
    """Loads the dataset and returns the concatenated version."""
    print("Loading dataset...")
    eval_dataset = load_dataset("webis/Touche23-ValueEval", revision="main")
    dataset = concatenate_datasets([eval_dataset["train"], eval_dataset["validation"], eval_dataset["test"]])
    print("Dataset loaded.")
    return dataset


def combine_argument(stance: str, conclusion: str, premise: str) -> str:
    """
    Combines stance, conclusion, and premise into a single string.
    """
    return f"{stance}: {conclusion}\nPremise: {premise}"


def create_neo4j_database(dataset: DatasetDict) -> None:
    """
    Creates a graph with nodes representing arguments.
    """
    print("Creating Neo4j database...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    for sample in tqdm(dataset, "Creating argument nodes"):
        combined_arg = combine_argument(sample[STANCE], sample[CONCLUSION], sample[PREMISE])
        node = Node(
            ARGUMENT,
            id=sample[ARGUMENT_ID],
            stance=sample[STANCE],
            premise=sample[PREMISE],
            conclusion=sample[CONCLUSION],
            labels=sample[LABELS],
            combined_argument=combined_arg
        )
        graph.create(node)

    for idx, label in tqdm(enumerate(labels), "Creating label nodes"):
        label_node = Node("LABEL", id=idx, name=label)
        graph.create(label_node)

    print("Neo4j database created.")


if __name__ == "__main__":
    dataset = load_data()
    create_neo4j_database(dataset)
