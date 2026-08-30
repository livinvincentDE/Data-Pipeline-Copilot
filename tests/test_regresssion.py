import pytest

from app.evaluate_regression import run_evaluation


@pytest.fixture(scope="module")
def evaluation_result():

    return run_evaluation(
        verbose=False
    )


def test_regression_dataset_not_empty(
    evaluation_result
):

    metrics = evaluation_result["metrics"]

    assert metrics["total_tests"] > 0


def test_regression_pass_rate(
    evaluation_result
):

    metrics = evaluation_result["metrics"]

    assert metrics["pass_rate"] == 1.0


def test_hit_rate_at_1(
    evaluation_result
):

    metrics = evaluation_result["metrics"]

    assert metrics["hit_rate_at_1"] >= 0.8


def test_hit_rate_at_3(
    evaluation_result
):

    metrics = evaluation_result["metrics"]

    assert metrics["hit_rate_at_3"] >= 0.9


def test_mrr_at_3(
    evaluation_result
):

    metrics = evaluation_result["metrics"]

    assert metrics["mrr_at_3"] >= 0.9