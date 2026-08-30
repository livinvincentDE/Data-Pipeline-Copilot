import json

from pathlib import Path
from datetime import datetime

from app.analytics import detect_technology


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# --------------------------------------------------
# FEEDBACK LOCATION
# --------------------------------------------------

FEEDBACK_DIR = (
    BASE_DIR
    / "data"
    / "feedback"
)

FEEDBACK_FILE = (
    FEEDBACK_DIR
    / "feedback.jsonl"
)


# --------------------------------------------------
# SAVE FEEDBACK
# --------------------------------------------------

def save_feedback(
    question,
    rewritten_question,
    answer,
    feedback,
    response_time=None,
    retrieved_document_count=None,
    query_rewriting_time=None,
    retrieval_time=None,
    reranking_time=None,
    llm_generation_time=None,
    retrieved_document_titles=None,
    retrieved_document_ids=None
):

    technology = detect_technology(question)

    FEEDBACK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    feedback_record = {

        "timestamp": datetime.now().isoformat(),

        "question": question,

        "rewritten_question": rewritten_question,

        "technology": technology,

        "answer": answer,

        "feedback": feedback,

        "response_time": response_time,

        "retrieved_document_count":
            retrieved_document_count,

        "query_rewriting_time":
            query_rewriting_time,

        "retrieval_time":
            retrieval_time,

        "reranking_time":
            reranking_time,

        "llm_generation_time":
            llm_generation_time,

        "retrieved_document_titles":
            retrieved_document_titles or [],

        "retrieved_document_ids":
            retrieved_document_ids or []
    }

    with open(
        FEEDBACK_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                feedback_record,
                ensure_ascii=False
            )
            + "\n"
        )


# --------------------------------------------------
# LOAD FEEDBACK
# --------------------------------------------------

def load_feedback():

    if not FEEDBACK_FILE.exists():
        return []

    feedback_records = []

    with open(
        FEEDBACK_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                feedback_records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:

                print(
                    "Warning: Skipping corrupted "
                    "feedback record."
                )

    return feedback_records