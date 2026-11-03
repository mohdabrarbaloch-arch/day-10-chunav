"""The AutoML engine: metrics, models, data handling and search.

Nothing in this package imports scikit-learn. Every estimator, every metric
and every split is implemented here so a score can be recomputed by hand.
"""

from .data import Dataset, load_dataset, profile_csv
from .search import run_search

__all__ = ["Dataset", "load_dataset", "profile_csv", "run_search"]
