from typing import Tuple

import numpy as np
from py2neo import Graph
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch

from neo4j.neo4j_config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME

EMBEDDINGS_FILENME = Path("bert_embeddings.npy")
DATA_FOLDER = Path("workspace/data/raw")


def load_bert() -> Tuple[AutoTokenizer, AutoModel]:
    """
    Loads the BERT model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = (
        AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        .to(device)
        .eval()
    )
    return tokenizer, model


def load_arguments_from_neo4j(
    database_uri: str, database_username: str, database_password: str
) -> list:
    """
    Loads the arguments from the Neo4j database.
    """
    graph = Graph(database_uri, auth=(database_username, database_password))
    query = """
    MATCH (a:ARGUMENT)
    RETURN a.combined_argument AS argument
    ORDER BY a.id
    """
    results = graph.run(query)
    return [result["argument"] for result in results]


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


def create_bert_embeddings(arguments: list) -> list[np.ndarray]:
    """
    Creates the BERT embeddings for the arguments.
    """
    bert_tokenizer, bert_model = load_bert()
    bert_embeddings = [
        get_bert_sentence_embeddings(argument, bert_tokenizer, bert_model)
        for argument in tqdm(arguments, desc="Creating BERT embeddings")
    ]
    return bert_embeddings


def save_bert_embeddings(bert_embeddings: np.ndarray, output_path: Path) -> None:
    """
    Saves the BERT embeddings to the given path.
    """
    if not DATA_FOLDER.exists():
        DATA_FOLDER.mkdir()
    np.save(DATA_FOLDER / output_path, bert_embeddings)


if __name__ == "__main__":
    arguments = load_arguments_from_neo4j(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    bert_embeddings = create_bert_embeddings(arguments)
    save_bert_embeddings(bert_embeddings, EMBEDDINGS_FILENME)
