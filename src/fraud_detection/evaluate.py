"""Metrics for a fraud model, shared by every model we train."""
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def evaluate(y_true, scores, threshold: float = 0.5) -> dict:
    """Score a model's fraud probabilities against the true labels.

    pr_auc / roc_auc judge the ranking at every threshold; precision / recall
    judge the yes/no decision at `threshold`.
    """
    predicted = scores >= threshold
    return {
        "pr_auc": average_precision_score(y_true, scores),        # main metric for rare fraud
        "roc_auc": roc_auc_score(y_true, scores),
        "precision": precision_score(y_true, predicted, zero_division=0),  # flagged -> really fraud
        "recall": recall_score(y_true, predicted),                # frauds -> actually flagged
    }


def threshold_for_recall(y_true, scores, target_recall: float = 0.9) -> float:
    """Highest threshold that still catches at least `target_recall` of the frauds.

    Highest = fewest false alarms among the thresholds that meet the recall target.
    Chosen on validation only, then applied unchanged to the test set.
    """
    _, recall, thresholds = precision_recall_curve(y_true, scores)
    # recall[i] is the recall when flagging scores >= thresholds[i]; it drops as the threshold rises
    meets_target = recall[:-1] >= target_recall
    return float(thresholds[meets_target].max())
