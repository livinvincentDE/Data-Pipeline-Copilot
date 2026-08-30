import numpy as np

from sentence_transformers import SentenceTransformer

from app.search import load_documents


# Load embedding model
# This model is lightweight and good for beginners
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def document_to_text(document):
    """
    Convert a document dictionary into text
    for embedding.
    """

    return (
        f"Technology: {document['technology']}. "
        f"Topic: {document['topic']}. "
        f"Title: {document['title']}. "
        f"Content: {document['content']}"
    )


def build_vector_index(documents):
    """
    Convert all documents into vector embeddings.
    """

    document_texts = [
        document_to_text(document)
        for document in documents
    ]

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True
    )

    return document_embeddings


def search_vector(
    query,
    documents,
    document_embeddings,
    top_k=3
):
    """
    Search documents using cosine similarity.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Because embeddings are normalized,
    # dot product = cosine similarity
    scores = np.dot(
        document_embeddings,
        query_embedding
    )

    scored_documents = []

    for document, score in zip(
        documents,
        scores
    ):

        scored_documents.append(
            {
                "document": document,
                "score": float(score)
            }
        )

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_documents[:top_k]


if __name__ == "__main__":

    print("Loading documents...")

    documents = load_documents()

    print(
        f"Building vector index for "
        f"{len(documents)} documents..."
    )

    document_embeddings = build_vector_index(
        documents
    )

    query = "Why is my Spark job slow?"

    results = search_vector(
        query=query,
        documents=documents,
        document_embeddings=document_embeddings,
        top_k=3
    )

    print("\nQuery:", query)

    print("\nTop Results:")

    for result in results:

        document = result["document"]

        print("\n-------------------------")

        print(
            "Score:",
            round(result["score"], 4)
        )

        print(
            "ID:",
            document["id"]
        )

        print(
            "Technology:",
            document["technology"]
        )

        print(
            "Title:",
            document["title"]
        )