from pathlib import Path

import numpy as np
import torch
from py2neo import Graph
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

ARGUMENTS_EMBEDDINGS_FILENME = Path("arguments_bert_embeddings.npy")
LABEL_EMBEDDINGS_FILENAME = Path("labels_bert_embeddings.npy")
DATA_FOLDER = Path("workspace/data/raw")


def load_bert() -> tuple[AutoTokenizer, AutoModel]:
    """
    Loads the BERT model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(device)
    return tokenizer, model


def get_neo4j_connection() -> Graph:
    """
    Returns a connection to the Neo4j database.
    """
    return Graph(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def load_arguments_from_neo4j(neo4j_connection: Graph) -> list:
    """
    Loads the arguments from the Neo4j database.
    """
    print("Loading arguments from Neo4j")

    query = """
    MATCH (a:ARGUMENT)
    RETURN a.combined_argument AS argument
    ORDER BY a.id
    """

    results = neo4j_connection.run(query)
    print("Arguments loaded from Neo4j")
    return [result["argument"] for result in results]


def load_labels_from_neo4j(neo4j_connection: Graph) -> list:
    """
    Loads the arguments from the Neo4j database.
    """
    print("Loading labels from Neo4j")

    query = """
    MATCH (l:LABEL)
    RETURN l.name as label
    ORDER BY l.id
    """

    results = neo4j_connection.run(query)
    print("Arguments loaded from Neo4j")
    return [result["label"] for result in results]


def get_bert_sentence_embeddings(
    text: str, bert_tokenizer: AutoTokenizer, bert_model: AutoModel
) -> np.ndarray:
    """
    Returns the BERT embeddings for the given text.
    """
    tokenized_text = bert_tokenizer(text, return_tensors="pt").to(bert_model.device)
    with torch.no_grad():
        outputs = bert_model(**tokenized_text)
    sentence_embedding = outputs.hidden_states[-2].mean(dim=1).squeeze().cpu().numpy()
    return sentence_embedding


def create_bert_embeddings(texts: list) -> list[np.ndarray]:
    """
    Creates the BERT embeddings for the given list of texts.
    """
    print("Creating BERT embeddings")
    bert_tokenizer, bert_model = load_bert()
    bert_embeddings = [
        get_bert_sentence_embeddings(text, bert_tokenizer, bert_model)
        for text in tqdm(texts, desc="Creating BERT embeddings")
    ]
    print("BERT embeddings created")
    return bert_embeddings


def save_bert_embeddings(bert_embeddings: np.ndarray, output_path: Path) -> None:
    """
    Saves the BERT embeddings to the given path.
    """
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
    np.save(DATA_FOLDER / output_path, bert_embeddings)
    print(f"BERT embeddings saved to {DATA_FOLDER / output_path}")


if __name__ == "__main__":
    neo4j_connection = get_neo4j_connection()
    arguments = load_arguments_from_neo4j(neo4j_connection)
    bert_embeddings = create_bert_embeddings(arguments)
    save_bert_embeddings(bert_embeddings, ARGUMENTS_EMBEDDINGS_FILENME)

    labels = load_labels_from_neo4j(neo4j_connection)
    label_bert_embeddings = create_bert_embeddings(labels)
    save_bert_embeddings(label_bert_embeddings, LABEL_EMBEDDINGS_FILENAME)
