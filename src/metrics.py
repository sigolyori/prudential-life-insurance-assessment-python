from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)


def qwk_score(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


qwk_scorer = make_scorer(qwk_score, greater_is_better=True)


def evaluate_multiclass(y_true, y_pred) -> Dict[str, Any]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "qwk": qwk_score(y_true, y_pred),
        "report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
