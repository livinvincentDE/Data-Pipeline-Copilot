import json
from pathlib import Path

from app.search import load_documents
from app.query_rewriter import rewrite_query
from app.bm25_search import build_bm25_index
from app.vector_search import build_vector_index
from app.hybrid_search import search_hybrid
from app.reranker import rerank_documents


# ==================================================
# PROJECT ROOT
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# REGRESSION DATASET
# ==================================================

REGRESSION_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "regression_questions.json"
)


# ==================================================
# LOAD REGRESSION QUESTIONS
# ==================================================

def load_regression_questions():

    if not REGRESSION_FILE.exists():

        raise FileNotFoundError(
            f"Regression dataset not found:\n"
            f"{REGRESSION_FILE}"
        )

    with open(
        REGRESSION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==================================================
# FIND DOCUMENT RANK
# ==================================================

def get_document_rank(
    expected_document_id,
    retrieved_document_ids
):

    for index, document_id in enumerate(
        retrieved_document_ids,
        start=1
    ):

        if document_id == expected_document_id:
            return index

    return None


# ==================================================
# RUN EVALUATION
# ==================================================

def run_evaluation(verbose=True):

    if verbose:

        print("\n" + "=" * 65)
        print("REGRESSION EVALUATION")
        print("=" * 65)


    # ----------------------------------------------
    # LOAD DOCUMENTS
    # ----------------------------------------------

    if verbose:
        print("\nLoading documents...")

    documents = load_documents()

    if verbose:
        print(f"Documents: {len(documents)}")


    # ----------------------------------------------
    # LOAD TEST DATASET
    # ----------------------------------------------

    regression_questions = (
        load_regression_questions()
    )

    if verbose:
        print(
            f"Regression tests: "
            f"{len(regression_questions)}"
        )


    # ----------------------------------------------
    # BUILD INDEXES
    # ----------------------------------------------

    if verbose:
        print("\nBuilding BM25 index...")

    bm25 = build_bm25_index(
        documents
    )


    if verbose:
        print("Building vector index...")

    document_embeddings = (
        build_vector_index(documents)
    )


    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    total_tests = 0

    hit_at_1 = 0

    hit_at_3 = 0

    reciprocal_rank_sum = 0

    failed_tests = []


    # ----------------------------------------------
    # RUN TESTS
    # ----------------------------------------------

    if verbose:

        print("\n" + "=" * 65)
        print("RUNNING REGRESSION TESTS")
        print("=" * 65)


    for test in regression_questions:

        question = test["question"]

        expected_document_id = (
            test["expected_document_id"]
        )


        # ------------------------------------------
        # QUERY REWRITING
        # ------------------------------------------

        rewritten_question = (
            rewrite_query(question)
        )


        # ------------------------------------------
        # HYBRID SEARCH
        # ------------------------------------------

        hybrid_results = search_hybrid(

            query=rewritten_question,

            documents=documents,

            bm25=bm25,

            document_embeddings=document_embeddings,

            top_k=10

        )


        # ------------------------------------------
        # RERANKING
        # ------------------------------------------

        reranked_results = rerank_documents(

            query=rewritten_question,

            search_results=hybrid_results,

            top_k=3

        )


        # ------------------------------------------
        # RETRIEVED DOCUMENT IDs
        # ------------------------------------------

        retrieved_document_ids = [

            result["document"].get("id")

            for result in reranked_results

        ]


        # ------------------------------------------
        # FIND RANK
        # ------------------------------------------

        rank = get_document_rank(

            expected_document_id,

            retrieved_document_ids

        )


        total_tests += 1


        # ------------------------------------------
        # HIT RATE @1
        # ------------------------------------------

        if rank == 1:
            hit_at_1 += 1


        # ------------------------------------------
        # HIT RATE @3 + MRR
        # ------------------------------------------

        if rank is not None:

            hit_at_3 += 1

            reciprocal_rank_sum += (
                1 / rank
            )


        # ------------------------------------------
        # DISPLAY
        # ------------------------------------------

        if verbose:

            print("\n" + "-" * 65)

            print(
                f"Test {total_tests}"
            )

            print(
                f"\nQuestion:\n{question}"
            )

            print(
                f"\nRewritten Query:\n"
                f"{rewritten_question}"
            )

            print(
                f"\nExpected Document:\n"
                f"{expected_document_id}"
            )

            print(
                f"\nRetrieved Documents:\n"
                f"{retrieved_document_ids}"
            )

            print(
                f"\nExpected Document Rank: "
                f"{rank if rank else 'Not Found'}"
            )

            if rank is not None:

                print("\nResult: PASS ✅")

            else:

                print("\nResult: FAIL ❌")


        # ------------------------------------------
        # STORE FAILURE
        # ------------------------------------------

        if rank is None:

            failed_tests.append({

                "question": question,

                "rewritten_question":
                    rewritten_question,

                "expected_document_id":
                    expected_document_id,

                "retrieved_document_ids":
                    retrieved_document_ids

            })


    # ==================================================
    # CALCULATE METRICS
    # ==================================================

    if total_tests > 0:

        hit_rate_at_1 = (
            hit_at_1 / total_tests
        )

        hit_rate_at_3 = (
            hit_at_3 / total_tests
        )

        mrr_at_3 = (
            reciprocal_rank_sum
            / total_tests
        )

        pass_rate = (
            hit_at_3
            / total_tests
        )

    else:

        hit_rate_at_1 = 0
        hit_rate_at_3 = 0
        mrr_at_3 = 0
        pass_rate = 0


    # ==================================================
    # RESULTS DICTIONARY
    # ==================================================

    metrics = {

        "total_tests": total_tests,

        "passed": hit_at_3,

        "failed": (
            total_tests
            - hit_at_3
        ),

        "pass_rate": round(
            pass_rate,
            4
        ),

        "hit_rate_at_1": round(
            hit_rate_at_1,
            4
        ),

        "hit_rate_at_3": round(
            hit_rate_at_3,
            4
        ),

        "mrr_at_3": round(
            mrr_at_3,
            4
        )

    }


    # ==================================================
    # PRINT FINAL RESULTS
    # ==================================================

    if verbose:

        print("\n" + "=" * 65)

        print(
            "REGRESSION EVALUATION RESULTS"
        )

        print("=" * 65)

        print()

        print(
            f"{'Metric':<30}"
            f"{'Score'}"
        )

        print("-" * 65)

        print(
            f"{'Total Tests':<30}"
            f"{metrics['total_tests']}"
        )

        print(
            f"{'Passed':<30}"
            f"{metrics['passed']}"
        )

        print(
            f"{'Failed':<30}"
            f"{metrics['failed']}"
        )

        print(
            f"{'Pass Rate':<30}"
            f"{metrics['pass_rate']:.1%}"
        )

        print(
            f"{'Hit Rate@1':<30}"
            f"{metrics['hit_rate_at_1']:.3f}"
        )

        print(
            f"{'Hit Rate@3':<30}"
            f"{metrics['hit_rate_at_3']:.3f}"
        )

        print(
            f"{'MRR@3':<30}"
            f"{metrics['mrr_at_3']:.3f}"
        )


        if failed_tests:

            print("\nFAILED TEST DETAILS")

            for failure in failed_tests:

                print("\n" + "-" * 65)

                print(
                    f"Question: "
                    f"{failure['question']}"
                )

                print(
                    f"Expected: "
                    f"{failure['expected_document_id']}"
                )

                print(
                    f"Retrieved: "
                    f"{failure['retrieved_document_ids']}"
                )

        else:

            print(
                "\n🎉 All regression tests passed!"
            )


    # ==================================================
    # RETURN RESULTS
    # ==================================================

    return {

        "metrics": metrics,

        "failed_tests": failed_tests

    }


# ==================================================
# RUN DIRECTLY
# ==================================================

if __name__ == "__main__":

    run_evaluation(verbose=True)