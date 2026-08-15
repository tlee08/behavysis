"""Diagnostics, EDA and improvement experiments for the behaviour classifier.

Each module is a runnable script
(``python -m behavysis.behaviour_classifier.eda.<name>``) that prints a
summary and writes a JSON report to ``data/front-rear/eda/``.
"""

__all__ = [
    "adversarial",
    "data_integrity",
    "experiments",
    "feature_quality",
    "sample_efficiency",
    "split_analysis",
]
