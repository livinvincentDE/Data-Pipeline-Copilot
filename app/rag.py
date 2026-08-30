import os
import time

from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from app.search import load_documents

from app.query_rewriter import rewrite_query

from app.bm25_search import build_bm25_index

from app.vector_search import build_vector_index

from app.hybrid_search import search_hybrid

from app.reranker import rerank_documents

from app.llm_prompts import build_basic_prompt


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv(
    BASE_DIR / ".env",
    override=True
)


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

api_key = os.getenv("GROQ_API_KEY")

if not api_key:

    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Check your .env file."
    )


client = Groq(
    api_key=api_key
)


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

print("Loading knowledge base...")

documents = load_documents()

print(
    f"Knowledge base loaded: "
    f"{len(documents)} documents"
)


# --------------------------------------------------
# BUILD INDEXES
# --------------------------------------------------

print("Building BM25 index...")

bm25 = build_bm25_index(
    documents
)


print("Building vector index...")

document_embeddings = build_vector_index(
    documents
)


# --------------------------------------------------
# BUILD CONTEXT
# --------------------------------------------------

def build_context(results):

    context_parts = []

    for result in results:

        document = result["document"]

        context_parts.append(

            f"""
Technology: {document.get("technology", "Unknown")}

Topic: {document.get("topic", "Unknown")}

Title: {document.get("title", "Unknown")}

Content:

{document.get("content", "")}
"""

        )

    return "\n\n---\n\n".join(
        context_parts
    )


# --------------------------------------------------
# RETRIEVE DOCUMENTS
# --------------------------------------------------

def retrieve_documents(
    query,
    retrieval_top_k=10,
    rerank_top_k=3
):

    # Hybrid Search

    hybrid_results = search_hybrid(

        query=query,

        documents=documents,

        bm25=bm25,

        document_embeddings=document_embeddings,

        top_k=retrieval_top_k

    )


    # Reranking

    reranked_results = rerank_documents(

        query=query,

        search_results=hybrid_results,

        top_k=rerank_top_k

    )


    return reranked_results


# --------------------------------------------------
# ANSWER QUESTION
# --------------------------------------------------

def answer_question(
    question,
    top_k=3
):

    total_start = time.perf_counter()


    # ==============================================
    # STEP 1: QUERY REWRITING
    # ==============================================

    rewrite_start = time.perf_counter()

    rewritten_question = rewrite_query(
        question
    )

    query_rewriting_time = (

        time.perf_counter()
        - rewrite_start

    )


    # ==============================================
    # STEP 2: HYBRID RETRIEVAL
    # ==============================================

    retrieval_start = time.perf_counter()

    hybrid_results = search_hybrid(

        query=rewritten_question,

        documents=documents,

        bm25=bm25,

        document_embeddings=document_embeddings,

        top_k=10

    )

    retrieval_time = (

        time.perf_counter()
        - retrieval_start

    )


    # ==============================================
    # STEP 3: RERANKING
    # ==============================================

    reranking_start = time.perf_counter()

    reranked_results = rerank_documents(

        query=rewritten_question,

        search_results=hybrid_results,

        top_k=top_k

    )

    reranking_time = (

        time.perf_counter()
        - reranking_start

    )


    # ==============================================
    # STEP 4: BUILD CONTEXT
    # ==============================================

    context = build_context(
        reranked_results
    )


    # ==============================================
    # STEP 5: BUILD PROMPT
    # ==============================================

    prompt = build_basic_prompt(

        question=question,

        context=context

    )


    # ==============================================
    # STEP 6: LLM GENERATION
    # ==============================================

    llm_start = time.perf_counter()

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


    answer = (

        response
        .choices[0]
        .message
        .content
        .strip()

    )


    llm_generation_time = (

        time.perf_counter()
        - llm_start

    )


    # ==============================================
    # TOTAL RESPONSE TIME
    # ==============================================

    response_time = (

        time.perf_counter()
        - total_start

    )


    # ==============================================
    # RETURN RESULTS
    # ==============================================

    return {

        "answer": answer,

        "original_question": question,

        "rewritten_question":
            rewritten_question,

        "documents":
            reranked_results,

        "retrieved_document_titles": [

            result["document"].get(
                "title",
                "Unknown"
            )

            for result in reranked_results

        ],

        "retrieved_document_ids": [

            result["document"].get(
                "id",
                "Unknown"
            )

            for result in reranked_results

        ],

        "context":
            context,

        "response_time":
            round(response_time, 3),

        "query_rewriting_time":
            round(query_rewriting_time, 3),

        "retrieval_time":
            round(retrieval_time, 3),

        "reranking_time":
            round(reranking_time, 3),

        "llm_generation_time":
            round(llm_generation_time, 3),

        "retrieved_document_count":
            len(reranked_results)
    }