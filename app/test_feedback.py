from app.feedback import (
    save_feedback,
    load_feedback
)


save_feedback(
    question="Why is my Spark job slow?",

    rewritten_question=(
        "Apache Spark slow performance troubleshooting"
    ),

    answer=(
        "Spark jobs can be slow because of data skew, "
        "shuffle operations, partitioning, or insufficient resources."
    ),

    feedback="positive"
)


feedback = load_feedback()


print("\nFEEDBACK RECORDS:\n")

for record in feedback:
    print(record)