import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

FEEDBACK_FILE = (
    BASE_DIR
    / "data"
    / "feedback"
    / "feedback.jsonl"
)

REGRESSION_FILE = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "regression_questions.json"
)


def load_negative_feedback():

    if not FEEDBACK_FILE.exists():

        print("No feedback file found.")

        return []


    negative_questions = []


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

                record = json.loads(line)

            except json.JSONDecodeError:

                continue


            if record.get("feedback") == "negative":

                negative_questions.append(
                    {

                        "question":
                            record.get(
                                "question"
                            ),

                        "technology":
                            record.get(
                                "technology",
                                "Unknown"
                            ),

                        "retrieved_document_ids":
                            record.get(
                                "retrieved_document_ids",
                                []
                            ),

                        "retrieved_document_titles":
                            record.get(
                                "retrieved_document_titles",
                                []
                            )

                    }
                )


    return negative_questions


def save_regression_questions(
    questions
):

    REGRESSION_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        REGRESSION_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            questions,

            file,

            indent=4,

            ensure_ascii=False

        )


def main():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "REGRESSION TEST DATASET BUILDER"
    )

    print(
        "=" * 60
    )


    questions = (
        load_negative_feedback()
    )


    print(
        f"\nNegative feedback questions: "
        f"{len(questions)}"
    )


    if not questions:

        print(
            "\nNo negative feedback available yet."
        )

        return


    save_regression_questions(
        questions
    )


    print(
        "\nRegression dataset created:"
    )

    print(
        REGRESSION_FILE
    )


if __name__ == "__main__":

    main()