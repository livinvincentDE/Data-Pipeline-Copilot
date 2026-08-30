import json

from pathlib import Path
from datetime import datetime

from app.evaluate_regression import run_evaluation


# ==================================================
# PROJECT ROOT
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ==================================================
# BASELINE FILE
# ==================================================

BASELINE_FILE = (

    BASE_DIR
    / "data"
    / "evaluation"
    / "baseline_results.json"

)


# ==================================================
# SAVE BASELINE
# ==================================================

def save_baseline(metrics):

    BASELINE_FILE.parent.mkdir(

        parents=True,
        exist_ok=True

    )


    baseline = {

        "created_at":

            datetime.now().isoformat(),

        "metrics":

            metrics

    }


    with open(

        BASELINE_FILE,
        "w",
        encoding="utf-8"

    ) as file:

        json.dump(

            baseline,

            file,

            indent=4

        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 65)

    print(
        "GENERATING REGRESSION BASELINE"
    )

    print("=" * 65)


    # ----------------------------------------------
    # RUN ACTUAL EVALUATION
    # ----------------------------------------------

    evaluation_result = (

        run_evaluation(
            verbose=False
        )

    )


    metrics = (

        evaluation_result[
            "metrics"
        ]

    )


    # ----------------------------------------------
    # SAVE BASELINE
    # ----------------------------------------------

    save_baseline(metrics)


    print(
        "\nBaseline successfully saved! 🎯"
    )

    print(
        f"\nLocation:\n"
        f"{BASELINE_FILE}"
    )


    print("\nBaseline Metrics:")

    print("-" * 40)


    for metric, value in metrics.items():

        print(
            f"{metric}: {value}"
        )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()