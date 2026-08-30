import json
from pathlib import Path

from app.search import load_documents, search

from app.bm25_search import (
    build_bm25_index,
    search_bm25,
)

from app.vector_search import (
    build_vector_index,
    search_vector,
)

from app.hybrid_search import search_hybrid

from app.reranker import rerank_documents


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "retrieval_eval.json"
)


# --------------------------------------------------
# LOAD EVALUATION DATA
# --------------------------------------------------

def load_evaluation_data():
    """
    Load evaluation questions and expected document IDs.
    """

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# --------------------------------------------------
# EXTRACT DOCUMENT IDS
# --------------------------------------------------

def get_retrieved_ids(results):
    """
    Extract document IDs from search results.
    """

    return [
        result["document"]["id"]
        for result in results
    ]


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(
    evaluation_data,
    retrieve_function,
    top_k=3
):
    """
    Calculate:

    - Hit Rate@K
    - MRR@K
    """

    hits = 0

    reciprocal_ranks = []

    for item in evaluation_data:

        question = item["question"]

        expected_id = (
            item["relevant_document_id"]
        )

        # Run retrieval
        results = retrieve_function(
            question,
            top_k
        )

        retrieved_ids = get_retrieved_ids(
            results
        )

        # Check if expected document was retrieved
        if expected_id in retrieved_ids:

            hits += 1

            rank = (
                retrieved_ids.index(expected_id)
                + 1
            )

            reciprocal_ranks.append(
                1 / rank
            )

        else:

            reciprocal_ranks.append(0)

    # Calculate Hit Rate
    hit_rate = (
        hits / len(evaluation_data)
    )

    # Calculate MRR
    mrr = (
        sum(reciprocal_ranks)
        / len(evaluation_data)
    )

    return hit_rate, mrr


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    # ----------------------------------------------
    # LOAD DOCUMENTS
    # ----------------------------------------------

    print("\nLoading documents...")

    documents = load_documents()

    evaluation_data = (
        load_evaluation_data()
    )

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Evaluation questions: "
        f"{len(evaluation_data)}"
    )

    # ----------------------------------------------
    # BUILD BM25 INDEX
    # ----------------------------------------------

    print("\nBuilding BM25 index...")

    bm25 = build_bm25_index(
        documents
    )

    # ----------------------------------------------
    # BUILD VECTOR INDEX
    # ----------------------------------------------

    print("Building vector index...")

    document_embeddings = (
        build_vector_index(
            documents
        )
    )

    # ==============================================
    # RETRIEVER 1
    # KEYWORD SEARCH
    # ==============================================

    def keyword_retriever(
        question,
        top_k
    ):

        return search(
            question,
            documents,
            top_k
        )

    # ==============================================
    # RETRIEVER 2
    # BM25 SEARCH
    # ==============================================

    def bm25_retriever(
        question,
        top_k
    ):

        return search_bm25(
            question,
            documents,
            bm25,
            top_k
        )

    # ==============================================
    # RETRIEVER 3
    # VECTOR SEARCH
    # ==============================================

    def vector_retriever(
        question,
        top_k
    ):

        return search_vector(
            question,
            documents,
            document_embeddings,
            top_k
        )

    # ==============================================
    # RETRIEVER 4
    # HYBRID SEARCH
    # ==============================================

    def hybrid_retriever(
        question,
        top_k
    ):

        return search_hybrid(
            question,
            documents,
            bm25,
            document_embeddings,
            top_k
        )

    # ==============================================
    # RETRIEVER 5
    # HYBRID SEARCH + RERANKER
    # ==============================================

    def hybrid_reranker_retriever(
        question,
        top_k
    ):

        # Step 1:
        # Retrieve more candidate documents
        hybrid_results = search_hybrid(
            query=question,
            documents=documents,
            bm25=bm25,
            document_embeddings=document_embeddings,
            top_k=10
        )

        # Step 2:
        # Rerank the candidate documents
        reranked_results = rerank_documents(
            query=question,
            search_results=hybrid_results,
            top_k=top_k
        )

        # Step 3:
        # Convert to standard result format
        return [
            {
                "document": result["document"],
                "score": result["reranker_score"]
            }
            for result in reranked_results
        ]

    # ----------------------------------------------
    # REGISTER RETRIEVERS
    # ----------------------------------------------

    retrievers = {

        "Keyword Search":
            keyword_retriever,

        "BM25":
            bm25_retriever,

        "Vector Search":
            vector_retriever,

        "Hybrid Search":
            hybrid_retriever,

        "Hybrid + Reranker":
            hybrid_reranker_retriever,
    }

    # ----------------------------------------------
    # EVALUATE
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    results_summary = []

    for name, retriever in retrievers.items():

        print(
            f"\nEvaluating: {name}"
        )

        hit_rate, mrr = calculate_metrics(
            evaluation_data,
            retriever,
            top_k=3
        )

        results_summary.append(
            {
                "name": name,
                "hit_rate": hit_rate,
                "mrr": mrr,
            }
        )

    # ----------------------------------------------
    # PRINT FINAL RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print()

    print(
        f"{'Retriever':<25}"
        f"{'Hit Rate@3':<15}"
        f"{'MRR@3':<10}"
    )

    print("-" * 50)

    for result in results_summary:

        print(
            f"{result['name']:<25}"
            f"{result['hit_rate']:<15.3f}"
            f"{result['mrr']:<10.3f}"
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()