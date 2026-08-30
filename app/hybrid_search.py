import numpy as np

from app.search import load_documents
from app.bm25_search import build_bm25_index
from app.vector_search import build_vector_index, model


def min_max_normalize(scores):
    """
    Normalize scores to a range between 0 and 1.
    """

    scores = np.array(scores)

    minimum = scores.min()
    maximum = scores.max()

    # Avoid division by zero
    if maximum == minimum:
        return np.ones(len(scores))

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


def search_hybrid(
    query,
    documents,
    bm25,
    document_embeddings,
    top_k=3,
    bm25_weight=0.5,
    vector_weight=0.5
):
    """
    Search using a combination of BM25
    and vector similarity.
    """

    # ---------------------------
    # BM25 SCORES
    # ---------------------------

    from app.search import tokenize

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(
        query_tokens
    )

    # ---------------------------
    # VECTOR SCORES
    # ---------------------------

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    vector_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    # ---------------------------
    # NORMALIZE SCORES
    # ---------------------------

    normalized_bm25 = min_max_normalize(
        bm25_scores
    )

    normalized_vector = min_max_normalize(
        vector_scores
    )

    # ---------------------------
    # COMBINE SCORES
    # ---------------------------

    hybrid_scores = (
        bm25_weight * normalized_bm25
        +
        vector_weight * normalized_vector
    )

    # ---------------------------
    # CREATE RESULTS
    # ---------------------------

    scored_documents = []

    for document, score in zip(
        documents,
        hybrid_scores
    ):

        scored_documents.append(
            {
                "document": document,
                "score": float(score)
            }
        )

    # Sort from highest score to lowest
    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_documents[:top_k]


if __name__ == "__main__":

    print("Loading documents...")

    documents = load_documents()

    print("Building BM25 index...")

    bm25 = build_bm25_index(
        documents
    )

    print("Building vector index...")

    document_embeddings = build_vector_index(
        documents
    )

    query = "Why is my Spark job slow?"

    results = search_hybrid(
        query=query,
        documents=documents,
        bm25=bm25,
        document_embeddings=document_embeddings,
        top_k=3
    )

    print("\nQuery:", query)

    print("\nTop Hybrid Results:")

    for result in results:

        document = result["document"]

        print("\n-------------------------")

        print(
            "Hybrid Score:",
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