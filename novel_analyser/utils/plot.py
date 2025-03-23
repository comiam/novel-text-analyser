"""
Функции для построения и сохранения графиков.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from novel_analyser.core.config import get_config
from novel_analyser.utils.stat import optimal_bins


def save_histogram(
    data: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    nbins: Optional[int] = None,
) -> None:
    """
    Строит и сохраняет гистограмму с оптимальным числом бинов.

    Args:
        data: Данные для гистограммы
        title: Заголовок графика
        xlabel: Название оси X
        ylabel: Название оси Y
        save_path: Путь для сохранения графика
        nbins: Число бинов (если None, определяется автоматически)
    """
    if nbins is None:
        nbins = optimal_bins(data)

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=nbins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_elbow_curve(
    inertias: List[float], max_clusters: int, save_path: str
) -> None:
    """
    Строит и сохраняет график elbow-curve для кластеризации.

    Args:
        inertias: Список инерций KMeans
        max_clusters: Максимальное число кластеров
        save_path: Путь для сохранения графика
    """
    ks: List[int] = list(range(1, max_clusters + 1))

    plt.figure(figsize=(8, 6))
    plt.plot(ks, inertias, marker="o")
    plt.title("Elbow Curve для KMeans кластеризации")
    plt.xlabel("Число кластеров (k)")
    plt.ylabel("Инерция")
    plt.grid(True)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_embedding_scatter(
    embeddings_2d: np.ndarray, labels: np.ndarray, title: str, save_path: str
) -> None:
    """
    Сохраняет диаграмму рассеяния для двумерных векторов эмбеддингов.

    Args:
        embeddings_2d: Двумерный массив эмбеддингов
        labels: Массив меток для цветовой кодировки точек
        title: Заголовок диаграммы
        save_path: Путь для сохранения изображения
    """
    plt.figure(figsize=(8, 6))

    if np.any(labels):
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
        )
    else:
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            color="blue",
            alpha=0.7,
        )

    plt.title(title)

    if np.any(labels):
        plt.colorbar(scatter)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_sentiment_histogram(sentiments: List[float], save_path: str) -> None:
    """
    Сохраняет гистограмму распределения значений сентимента в указанный файл.

    Args:
        sentiments: Список значений сентимента
        save_path: Путь к файлу, в который будет сохранена гистограмма
    """
    nbins: int = optimal_bins(sentiments)

    plt.figure(figsize=(8, 6))
    plt.hist(sentiments, bins=nbins, edgecolor="black")

    plt.title("Распределение эмоционального окраса блоков")
    plt.xlabel("Сентимент (полярность)")
    plt.ylabel("Количество блоков")

    plt.grid(True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_sentiment_pie_chart(
    sentiments: List[float],
    save_path: str,
    positive_threshold: Optional[float] = None,
    negative_threshold: Optional[float] = None,
) -> None:
    """
    Сохраняет круговую диаграмму распределения эмоционального окраса блоков.

    Args:
        sentiments: Список значений эмоционального окраса
        save_path: Путь для сохранения изображения диаграммы
        positive_threshold: Порог для определения положительного настроения
        negative_threshold: Порог для определения отрицательного настроения
    """
    if positive_threshold is None or negative_threshold is None:
        config = get_config()

        if positive_threshold is None:
            positive_threshold = config.sentiment_analyze.positive_threshold
        if negative_threshold is None:
            negative_threshold = config.sentiment_analyze.negative_threshold

    pos: int = sum(1 for s in sentiments if s > positive_threshold)
    neg: int = sum(1 for s in sentiments if s < negative_threshold)
    neu: int = len(sentiments) - pos - neg

    labels: List[str] = ["Положительные", "Нейтральные", "Отрицательные"]
    sizes: List[int] = [pos, neu, neg]

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)

    plt.title("Распределение эмоционального окраса блоков")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_bar_chart(
    labels: List[str],
    values: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
    filter_small_values: bool = True,
) -> None:
    """
    Создает и сохраняет столбчатую диаграмму.

    Args:
        labels: Метки для элементов диаграммы
        values: Значения для диаграммы
        title: Заголовок диаграммы
        xlabel: Название оси X
        ylabel: Название оси Y
        save_path: Путь для сохранения изображения
        filter_small_values: Фильтровать ли незначимые значения
    """
    from novel_analyser.utils.stat import filter_insignificant_values

    if filter_small_values:
        filtered_labels, filtered_values = filter_insignificant_values(
            labels, values
        )
    else:
        filtered_labels, filtered_values = labels, values

    # Сортируем по значениям для лучшей визуализации
    if filtered_labels:
        sorted_data = sorted(
            zip(filtered_labels, filtered_values),
            key=lambda x: x[1],
            reverse=True,
        )
        filtered_labels, filtered_values = zip(*sorted_data)
        filtered_labels, filtered_values = list(filtered_labels), list(
            filtered_values
        )

    plt.figure(figsize=(10, 6))
    plt.bar(filtered_labels, filtered_values)

    if filter_small_values and len(filtered_labels) < len(labels):
        filtered_out = len(labels) - len(filtered_labels)
        title += f" (отфильтровано {filtered_out} незначимых значений)"

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
