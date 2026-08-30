from rank_bm25 import BM25Okapi

from app.search import load_documents, tokenize


def build_bm25_index(documents):
    """
    Build a BM25 index from the knowledge base documents.
    """

    tokenized_documents = []

    for document in documents:

        text = (
            document["technology"]
            + " "
            + document["topic"]
            + " "
            + document["title"]
            + " "
            + document["content"]
        )

        tokens = tokenize(text)

        tokenized_documents.append(tokens)

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def search_bm25(query, documents, bm25, top_k=3):
    """
    Search documents using BM25.
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    scored_documents = []

    for document, score in zip(documents, scores):

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

    documents = load_documents()

    bm25 = build_bm25_index(documents)

    query = "Why is my Spark job slow?"

    results = search_bm25(
        query=query,
        documents=documents,
        bm25=bm25,
        top_k=3
    )

    print("\nQuery:", query)

    print("\nTop Results:")

    for result in results:

        document = result["document"]

        print("\n-------------------------")

        print("Score:", round(result["score"], 4))

        print("ID:", document["id"])

        print("Technology:", document["technology"])

        print("Title:", document["title"])