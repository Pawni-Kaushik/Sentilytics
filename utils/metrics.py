"""
utils/metrics.py
------------------
Convenience accessors around the precomputed metrics JSON (generated
once from the trained model + dataset -- see the training notebook).
We precompute rather than recompute on every page load because
re-running inference over the full test set on every Streamlit rerun
would be slow and pointless (the model isn't changing at runtime).
"""

import pandas as pd

from utils.loader import load_precomputed_metrics


def get_classification_report_df() -> pd.DataFrame:
    metrics = load_precomputed_metrics()
    report = metrics["classification_report"]

    rows = []
    for label, stats in report.items():
        if label in ("accuracy",):
            continue
        if not isinstance(stats, dict):
            continue
        rows.append({
            "Class": label,
            "Precision": round(stats["precision"], 3),
            "Recall": round(stats["recall"], 3),
            "F1-Score": round(stats["f1-score"], 3),
            "Support": int(stats["support"]),
        })
    return pd.DataFrame(rows)


def get_confusion_matrix():
    metrics = load_precomputed_metrics()
    return metrics["confusion_matrix"], metrics["labels"]


def get_training_history():
    metrics = load_precomputed_metrics()
    return metrics["training_history"]


def get_roc_data():
    metrics = load_precomputed_metrics()
    return metrics["roc_curve"]


def get_dataset_stats():
    metrics = load_precomputed_metrics()
    return metrics["dataset_stats"]


def get_headline_numbers():
    metrics = load_precomputed_metrics()
    return {
        "test_accuracy": metrics["test_accuracy"],
        "vocab_size": metrics["vocab_size"],
        "total_training_rows": metrics["total_training_rows"],
    }


def get_ner_accuracy():
    """
    Reads the NER model's test accuracy from ner_config.json. Returns None
    if the NER artifacts aren't available yet, so callers can fail gracefully
    (e.g. show a placeholder) instead of crashing.
    """
    import json
    from config import NER_CONFIG_PATH
    try:
        with open(NER_CONFIG_PATH, "r") as f:
            ner_config = json.load(f)
        return ner_config.get("test_accuracy")
    except Exception:
        return None
