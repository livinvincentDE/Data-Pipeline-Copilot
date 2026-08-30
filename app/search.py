import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "documents.json"
)


def load_documents():

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    return documents


def tokenize(text):
    """
    Convert text into lowercase words
    and remove punctuation.
    """

    return re.findall(r"\b\w+\b", text.lower())


def search(query, documents, top_k=3):

    query_words = tokenize(query)

    scored_documents = []

    for document in documents:

        text = (
            document["title"]
            + " "
            + document["content"]
        )

        document_words = tokenize(text)

        score = 0

        for word in query_words:

            if word in document_words:
                score += 1

        scored_documents.append(
            {
                "document": document,
                "score": score
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_documents[:top_k]