from sentence_transformers import CrossEncoder


# Cross-Encoder model for document reranking
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


print("Loading reranking model...")

reranker_model = CrossEncoder(MODEL_NAME)


def document_to_text(document):
    """
    Convert a document into text for reranking.
    """

    return (
        f"Technology: {document['technology']}. "
        f"Topic: {document['topic']}. "
        f"Title: {document['title']}. "
        f"Content: {document['content']}"
    )


def rerank_documents(query, search_results, top_k=3):
    """
    Rerank retrieved documents using a Cross-Encoder.

    Parameters:
        query: User search query
        search_results: Results returned by the retriever
        top_k: Number of final documents to return
    """

    # Create query-document pairs
    pairs = []

    for result in search_results:

        document = result["document"]

        document_text = document_to_text(document)

        pairs.append(
            (query, document_text)
        )

    # Get reranking scores
    scores = reranker_model.predict(pairs)

    reranked_results = []

    for result, score in zip(
        search_results,
        scores
    ):

        reranked_results.append(
            {
                "document": result["document"],
                "retrieval_score": result["score"],
                "reranker_score": float(score)
            }
        )

    # Sort by reranker score
    reranked_results.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]