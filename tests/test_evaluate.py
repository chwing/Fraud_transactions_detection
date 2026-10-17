import numpy as np
import pytest

from fraud_detection.evaluate import evaluate, threshold_for_recall


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    metrics = evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics == {"pr_auc": 1.0, "roc_auc": 1.0, "precision": 1.0, "recall": 1.0}


def test_precision_and_recall_at_threshold():
    y = np.array([0, 0, 1, 1])
    # At 0.5: flags rows 2 and 3 -> one real fraud, one false alarm; one fraud missed
    metrics = evaluate(y, np.array([0.1, 0.6, 0.4, 0.9]))
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)


def test_threshold_for_recall():
    # 4 frauds with scores 0.9, 0.8, 0.7, 0.3; legit rows at 0.75 and 0.1
    y = np.array([1, 1, 1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.3, 0.75, 0.1])

    assert threshold_for_recall(y, scores, target_recall=0.75) == 0.7   # catches 3 of 4
    assert threshold_for_recall(y, scores, target_recall=1.0) == 0.3    # must go down to catch all
