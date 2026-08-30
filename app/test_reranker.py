from app.search import load_documents
from app.bm25_search import build_bm25_index
from app.vector_search import build_vector_index
from app.hybrid_search import search_hybrid
from app.reranker import rerank_documents


def main():

    print("Loading documents...")

    documents = load_documents()

    print("Building BM25 index...")

    bm25 = build_bm25_index(documents)

    print("Building vector index...")

    document_embeddings = build_vector_index(documents)

    query = "Why is my Spark job slow?"

    # Retrieve more documents before reranking
    search_results = search_hybrid(
        query=query,
        documents=documents,
        bm25=bm25,
        document_embeddings=document_embeddings,
        top_k=10
    )

    print("\n" + "=" * 60)
    print("BEFORE RERANKING")
    print("=" * 60)

    for rank, result in enumerate(search_results, start=1):

        document = result["document"]

        print(
            f"{rank}. "
            f"{document['id']} | "
            f"{document['title']} | "
            f"Score: {result['score']:.4f}"
        )

    # Rerank the retrieved documents
    reranked_results = rerank_documents(
        query=query,
        search_results=search_results,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("AFTER RERANKING")
    print("=" * 60)

    for rank, result in enumerate(reranked_results, start=1):

        document = result["document"]

        print(
            f"{rank}. "
            f"{document['id']} | "
            f"{document['title']}"
        )

        print(
            f"   Retrieval Score: "
            f"{result['retrieval_score']:.4f}"
        )

        print(
            f"   Reranker Score: "
            f"{result['reranker_score']:.4f}"
        )


if __name__ == "__main__":
    main()