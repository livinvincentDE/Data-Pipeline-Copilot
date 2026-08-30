import json

from pathlib import Path

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
# LOAD BASELINE
# ==================================================

def load_baseline():

    if not BASELINE_FILE.exists():

        raise FileNotFoundError(

            "Baseline file not found:\n"
            f"{BASELINE_FILE}\n\n"
            "Run this first:\n"
            "python -m app.save_baseline"

        )

    with open(
        BASELINE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        baseline = json.load(file)

    return baseline


# ==================================================
# COMPARE METRICS
# ==================================================

def compare_metrics(
    baseline_metrics,
    current_metrics
):

    metrics_to_compare = [

        "pass_rate",

        "hit_rate_at_1",

        "hit_rate_at_3",

        "mrr_at_3"

    ]


    comparison_results = []

    regression_detected = False


    for metric in metrics_to_compare:

        baseline_value = (
            baseline_metrics.get(metric, 0)
        )

        current_value = (
            current_metrics.get(metric, 0)
        )


        difference = (

            current_value
            - baseline_value

        )


        # ------------------------------------------
        # DETERMINE STATUS
        # ------------------------------------------

        if difference < 0:

            status = "REGRESSION ❌"

            regression_detected = True


        elif difference > 0:

            status = "IMPROVED ✅"


        else:

            status = "UNCHANGED ➖"


        comparison_results.append({

            "metric": metric,

            "baseline": baseline_value,

            "current": current_value,

            "difference": difference,

            "status": status

        })


    return (

        comparison_results,

        regression_detected

    )


# ==================================================
# DISPLAY COMPARISON
# ==================================================

def display_comparison(
    comparison_results
):

    print("\n")

    print(
        f"{'Metric':<20}"
        f"{'Baseline':>12}"
        f"{'Current':>12}"
        f"{'Change':>12}"
        f"{'Status':>20}"
    )

    print(
        "-" * 80
    )


    for result in comparison_results:

        metric = result["metric"]

        baseline = result["baseline"]

        current = result["current"]

        difference = result["difference"]

        status = result["status"]


        print(

            f"{metric:<20}"

            f"{baseline:>12.3f}"

            f"{current:>12.3f}"

            f"{difference:>+12.3f}"

            f"{status:>20}"

        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n" + "=" * 80)

    print(
        "RAG PIPELINE BASELINE COMPARISON"
    )

    print("=" * 80)


    # ----------------------------------------------
    # LOAD BASELINE
    # ----------------------------------------------

    print(
        "\nLoading baseline..."
    )

    baseline = load_baseline()


    baseline_metrics = (
        baseline["metrics"]
    )


    print(
        "Baseline created:"
    )

    print(
        baseline.get(
            "created_at",
            "Unknown"
        )
    )


    # ----------------------------------------------
    # RUN CURRENT EVALUATION
    # ----------------------------------------------

    print(
        "\nRunning current evaluation..."
    )


    evaluation_result = (

        run_evaluation(
            verbose=False
        )

    )


    current_metrics = (

        evaluation_result[
            "metrics"
        ]

    )


    # ----------------------------------------------
    # COMPARE RESULTS
    # ----------------------------------------------

    comparison_results, regression_detected = (

        compare_metrics(

            baseline_metrics,

            current_metrics

        )

    )


    # ----------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------

    print("\n" + "=" * 80)

    print(
        "BASELINE VS CURRENT"
    )

    print("=" * 80)


    display_comparison(
        comparison_results
    )


    # ==================================================
    # FINAL STATUS
    # ==================================================

    print("\n" + "=" * 80)

    print(
        "FINAL STATUS"
    )

    print("=" * 80)


    if regression_detected:

        print(

            "\n❌ REGRESSION DETECTED"

        )

        print(

            "\nOne or more evaluation "
            "metrics decreased compared "
            "with the baseline."

        )


        print(

            "\nReview recent changes to:"

        )

        print(
            "- Query rewriting"
        )

        print(
            "- Hybrid search"
        )

        print(
            "- BM25 weights"
        )

        print(
            "- Vector search"
        )

        print(
            "- Reranking"
        )


    else:

        print(

            "\n✅ NO REGRESSION DETECTED"

        )

        print(

            "\nCurrent pipeline performance "
            "matches or exceeds the baseline."

        )


    print("\n" + "=" * 80)


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()