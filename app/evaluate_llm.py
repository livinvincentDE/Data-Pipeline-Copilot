import json
from pathlib import Path

from app.search import load_documents

from app.bm25_search import build_bm25_index

from app.vector_search import build_vector_index

from app.hybrid_search import search_hybrid

from app.reranker import rerank_documents

from app.llm_prompts import (
    build_basic_prompt,
    build_improved_prompt,
)

from app.rag import client


BASE_DIR = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "llm_eval.json"
)


def load_evaluation_data():

    with open(
        EVALUATION_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def build_context(results):

    context_parts = []

    for result in results:

        document = result["document"]

        context_parts.append(
            f"""
Technology: {document["technology"]}

Title: {document["title"]}

Content:
{document["content"]}
"""
        )

    return "\n\n---\n\n".join(context_parts)


def retrieve_documents(
    question,
    documents,
    bm25,
    document_embeddings
):

    # Hybrid retrieval
    hybrid_results = search_hybrid(
        query=question,
        documents=documents,
        bm25=bm25,
        document_embeddings=document_embeddings,
        top_k=10
    )

    # Reranking
    reranked_results = rerank_documents(
        query=question,
        search_results=hybrid_results,
        top_k=3
    )

    return reranked_results


def call_llm(prompt):

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return (
        response
        .choices[0]
        .message
        .content
        .strip()
    )


def evaluate_answer(
    answer,
    expected_topics
):
    """
    Simple topic coverage evaluation.

    Score = percentage of expected topics
    mentioned in the answer.
    """

    answer_lower = answer.lower()

    topics_found = []

    for topic in expected_topics:

        if topic.lower() in answer_lower:

            topics_found.append(topic)

    score = (
        len(topics_found)
        / len(expected_topics)
    )

    return score, topics_found


def main():

    print("=" * 65)
    print("LLM ANSWER EVALUATION")
    print("=" * 65)

    # ------------------------------------------
    # LOAD DATA
    # ------------------------------------------

    documents = load_documents()

    evaluation_data = load_evaluation_data()

    print(
        f"\nDocuments: {len(documents)}"
    )

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

    document_embeddings = (
        build_vector_index(documents)
    )

    # ------------------------------------------
    # STORE SCORES
    # ------------------------------------------

    basic_scores = []

    improved_scores = []

    # ==========================================
    # EVALUATE QUESTIONS
    # ==========================================

    for index, item in enumerate(
        evaluation_data,
        start=1
    ):

        question = item["question"]

        expected_topics = (
            item["expected_topics"]
        )

        print("\n" + "=" * 65)

        print(
            f"QUESTION {index}"
        )

        print("=" * 65)

        print(question)

        # --------------------------------------
        # RETRIEVE CONTEXT
        # --------------------------------------

        results = retrieve_documents(
            question,
            documents,
            bm25,
            document_embeddings
        )

        context = build_context(
            results
        )

        # --------------------------------------
        # BASIC PROMPT
        # --------------------------------------

        basic_prompt = build_basic_prompt(
            question,
            context
        )

        basic_answer = call_llm(
            basic_prompt
        )

        basic_score, basic_topics = (
            evaluate_answer(
                basic_answer,
                expected_topics
            )
        )

        basic_scores.append(
            basic_score
        )

        # --------------------------------------
        # IMPROVED PROMPT
        # --------------------------------------

        improved_prompt = build_improved_prompt(
            question,
            context
        )

        improved_answer = call_llm(
            improved_prompt
        )

        improved_score, improved_topics = (
            evaluate_answer(
                improved_answer,
                expected_topics
            )
        )

        improved_scores.append(
            improved_score
        )

        # --------------------------------------
        # PRINT RESULTS
        # --------------------------------------

        print("\nBASIC ANSWER:")

        print(basic_answer)

        print(
            f"\nScore: {basic_score:.2f}"
        )

        print(
            f"Topics Found: {basic_topics}"
        )

        print("\n" + "-" * 65)

        print("\nIMPROVED ANSWER:")

        print(improved_answer)

        print(
            f"\nScore: "
            f"{improved_score:.2f}"
        )

        print(
            f"Topics Found: "
            f"{improved_topics}"
        )

    # ==========================================
    # FINAL RESULTS
    # ==========================================

    average_basic = (
        sum(basic_scores)
        / len(basic_scores)
    )

    average_improved = (
        sum(improved_scores)
        / len(improved_scores)
    )

    print("\n" + "=" * 65)

    print("FINAL LLM EVALUATION RESULTS")

    print("=" * 65)

    print()

    print(
        f"{'Prompt Strategy':<30}"
        f"{'Average Score':<15}"
    )

    print("-" * 45)

    print(
        f"{'Basic Prompt':<30}"
        f"{average_basic:.3f}"
    )

    print(
        f"{'Improved Prompt':<30}"
        f"{average_improved:.3f}"
    )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()