"""
Различные утилиты для библиотеки анализа текста.
"""

from novel_analyser.utils.plot import (
    save_histogram,
    save_elbow_curve,
    save_embedding_scatter,
)
from novel_analyser.utils.stat import (
    filter_outliers_by_percentiles,
    optimal_bins,
    compute_average_cosine_similarity,
)
from novel_analyser.utils.stopwords import get_stop_words

__all__ = [
    "get_stop_words",
    "filter_outliers_by_percentiles",
    "optimal_bins",
    "compute_average_cosine_similarity",
    "save_histogram",
    "save_elbow_curve",
    "save_embedding_scatter",
]
