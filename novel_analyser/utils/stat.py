"""
Статистические утилиты для анализа данных.
"""

from math import ceil
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

from novel_analyser.core.config import get_config


def optimal_bins(data: List[float]) -> int:
    """
    Определяет оптимальное число бинов для гистограммы по правилу Фридмана–Диаконса.

    Args:
        data: Список числовых данных

    Returns:
        Оптимальное число бинов
    """
    n = len(data)
    if n < 2:
        return 1

    config = get_config()
    data_arr = np.array(data)

    q1 = np.percentile(data_arr, config.statistics.quartile_first)
    q3 = np.percentile(data_arr, config.statistics.quartile_third)
    iqr = q3 - q1

    # Если межквартильный размах равен 0, используем правило Стёрджесса
    if iqr == 0:
        return int(ceil(np.log2(n) + config.statistics.sturges_offset))

    data_range = np.max(data_arr) - np.min(data_arr)
    if data_range == 0:
        return 1

    bin_width = (
        config.statistics.friedman_diaconis_factor * iqr / (n ** (1 / 3))
    )
    if bin_width == 0:
        return 1

    # Ограничиваем количество бинов разумными пределами
    bins = max(int(ceil(data_range / bin_width)), 1)
    return max(
        min(bins, config.statistics.histogram_bins_max),
        config.statistics.histogram_bins_min,
    )


def filter_outliers_by_percentiles(
    data: List[float],
    lower_percentile: Optional[float] = None,
    upper_percentile: Optional[float] = None,
) -> List[float]:
    """
    Фильтрует выбросы из данных на основе перцентилей.

    Args:
        data: Список числовых данных
        lower_percentile: Нижний перцентиль
        upper_percentile: Верхний перцентиль

    Returns:
        Отфильтрованные данные без выбросов
    """
    if lower_percentile is None or upper_percentile is None:
        config = get_config()
        if lower_percentile is None:
            lower_percentile = config.statistics.filter_percentile_lower
        if upper_percentile is None:
            upper_percentile = config.statistics.filter_percentile_upper

    if not data or len(data) < 2:
        return data.copy()

    data_arr = np.array(data)

    lower_bound = np.percentile(data_arr, lower_percentile)
    upper_bound = np.percentile(data_arr, upper_percentile)

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    return filtered_data


def compute_average_cosine_similarity(embeddings: np.ndarray) -> float:
    """
    Вычисляет среднюю косинусную близость между соседними эмбеддингами.

    Args:
        embeddings: Массив эмбеддингов

    Returns:
        Средняя косинусная близость
    """
    if len(embeddings) <= 1:
        return 0.0

    config = get_config()
    threshold = config.statistics.cosine_similarity_threshold
    sims: List[float] = []

    for i in range(len(embeddings) - 1):
        v1 = embeddings[i]
        v2 = embeddings[i + 1]

        cos_sim: float = np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-9)
        # Ограничиваем значения косинусной близости порогом из конфигурации
        if cos_sim > threshold:
            sims.append(cos_sim)
        else:
            sims.append(0.0)

    return float(np.mean(sims))


def filter_insignificant_values(
    labels: List[str], values: List[float], iqr_factor: Optional[float] = None
) -> Tuple[List[str], List[float]]:
    """
    Фильтрует статистически незначимые (близкие к нулю) значения используя метод
    межквартильного размаха (IQR).

    Args:
        labels: Список меток для значений
        values: Список значений
        iqr_factor: Множитель для IQR (если None, берется из конфигурации)

    Returns:
        Отфильтрованные списки меток и значений
    """
    if len(values) < 4:  # Для малых наборов данных фильтрация не применяется
        return labels, values

    # Получаем множитель IQR и другие параметры из конфигурации
    config = get_config()
    if iqr_factor is None:
        iqr_factor = config.statistics.iqr_factor

    # Расчет квартилей и IQR на основе абсолютных значений
    abs_values = np.abs(values)
    q1 = np.percentile(abs_values, config.statistics.quartile_first)
    q3 = np.percentile(abs_values, config.statistics.quartile_third)
    iqr = q3 - q1

    # Нижняя граница для значимых значений
    lower_bound = q1 + iqr_factor * iqr

    # Фильтрация значений, близких к нулю
    filtered_data = [
        (char, val)
        for char, val in zip(labels, values)
        if abs(val) >= lower_bound
    ]

    if not filtered_data:  # Если все отфильтровали, возвращаем исходные данные
        return labels, values

    filtered_chars, filtered_values = zip(*filtered_data)
    return list(filtered_chars), list(filtered_values)
