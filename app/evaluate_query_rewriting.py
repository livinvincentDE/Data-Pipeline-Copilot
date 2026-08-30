from app.search import load_documents

from app.bm25_search import build_bm25_index

from app.vector_search import build_vector_index

from app.hybrid_search import search_hybrid

from app.reranker import rerank_documents

from app.query_rewriter import rewrite_query

from app.evaluate_retrieval import (
    load_evaluation_data,
    calculate_metrics,
)


def main():

    print("=" * 65)
    print("QUERY REWRITING EVALUATION")
    print("=" * 65)

    # ------------------------------------------
    # LOAD DOCUMENTS
    # ------------------------------------------

    print("\nLoading documents...")

    documents = load_documents()

    evaluation_data = load_evaluation_data()

    print(f"Documents: {len(documents)}")

    print(
        f"Evaluation questions: "
        f"{len(evaluation_data)}"
    )

    # ------------------------------------------
    # BUILD RETRIEVAL INDEXES
    # ------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25_index(
        documents
    )

    print("Building vector index...")

    document_embeddings = build_vector_index(
        documents
    )

    # ------------------------------------------
    # ORIGINAL QUERY RETRIEVER
    # ------------------------------------------

    def original_query_retriever(
        question,
        top_k
    ):

        # Step 1: Hybrid retrieval
        hybrid_results = search_hybrid(
            query=question,
            documents=documents,
            bm25=bm25,
            document_embeddings=document_embeddings,
            top_k=10
        )

        # Step 2: Rerank
        reranked_results = rerank_documents(
            query=question,
            search_results=hybrid_results,
            top_k=top_k
        )

        # Convert to standard format
        return [
            {
                "document": result["document"],
                "score": result["reranker_score"]
            }
            for result in reranked_results
        ]

    # ------------------------------------------
    # REWRITTEN QUERY RETRIEVER
    # ------------------------------------------

    def rewritten_query_retriever(
        question,
        top_k
    ):

        # Step 1: Rewrite query with Groq
        rewritten_query = rewrite_query(
            question
        )

        # Step 2: Hybrid retrieval
        hybrid_results = search_hybrid(
            query=rewritten_query,
            documents=documents,
            bm25=bm25,
            document_embeddings=document_embeddings,
            top_k=10
        )

        # Step 3: Rerank
        reranked_results = rerank_documents(
            query=rewritten_query,
            search_results=hybrid_results,
            top_k=top_k
        )

        # Convert to standard format
        return [
            {
                "document": result["document"],
                "score": result["reranker_score"]
            }
            for result in reranked_results
        ]

    # ==========================================
    # SHOW EXAMPLES
    # ==========================================

    print("\n" + "=" * 65)
    print("QUERY REWRITING EXAMPLES")
    print("=" * 65)

    for item in evaluation_data[:5]:

        original_question = item["question"]

        rewritten_question = rewrite_query(
            original_question
        )

        print("\nOriginal:")
        print(original_question)

        print("\nRewritten:")
        print(rewritten_question)

        print("-" * 65)

    # ==========================================
    # EVALUATE ORIGINAL QUERIES
    # ==========================================

    print("\nEvaluating original queries...")

    original_hit_rate, original_mrr = (
        calculate_metrics(
            evaluation_data,
            original_query_retriever,
            top_k=3
        )
    )

    # ==========================================
    # EVALUATE REWRITTEN QUERIES
    # ==========================================

    print("\nEvaluating rewritten queries...")

    rewritten_hit_rate, rewritten_mrr = (
        calculate_metrics(
            evaluation_data,
            rewritten_query_retriever,
            top_k=3
        )
    )

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    print()

    print(
        f"{'Approach':<30}"
        f"{'Hit Rate@3':<15}"
        f"{'MRR@3':<10}"
    )

    print("-" * 55)

    print(
        f"{'Original Query':<30}"
        f"{original_hit_rate:<15.3f}"
        f"{original_mrr:<10.3f}"
    )

    print(
        f"{'Groq Rewritten Query':<30}"
        f"{rewritten_hit_rate:<15.3f}"
        f"{rewritten_mrr:<10.3f}"
    )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()