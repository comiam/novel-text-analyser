"""
Модуль для тематического моделирования текста.
"""

import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.utils.stopwords import get_stop_words


class TopicAnalyser(BaseAnalyser):
    """
    Класс для тематического моделирования текста с использованием LDA.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует анализатор тем.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.vectorizer = CountVectorizer(stop_words=get_stop_words())

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет тематическое моделирование текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Находим оптимальное количество тем
        optimal_n, topic_keywords, perplexities = (
            self.find_optimal_number_of_topics(
                blocks,
                min_topics=self.config.analyse.min_topics,
                max_topics=self.config.analyse.max_topics,
            )
        )

        # Выполняем тематическое моделирование с оптимальным числом тем
        topic_percentages = self.perform_topic_modeling(
            blocks, num_topics=optimal_n
        )

        # Сохраняем графики
        self.save_optimal_topics_plots(
            perplexities,
            self.config.analyse.min_topics,
            self.config.analyse.max_topics,
            topic_percentages,
            topic_keywords,
        )

        # Заполняем метрики
        result.metrics.update(
            {
                "optimal_number_of_topics": optimal_n,
                "topic_keywords": topic_keywords,
                "topic_percentages": topic_percentages,
                "perplexities": perplexities,
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "perplexity_plot": os.path.join(
                    self.config.output.output_dir,
                    "optimal_topics_perplexity.png",
                ),
                "topic_distribution": os.path.join(
                    self.config.output.output_dir, "topic_distribution_bar.png"
                ),
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Тематическое моделирование:\n"
        result.summary += f"  Оптимальное число тем: {optimal_n}\n"

        for idx, words in topic_keywords.items():
            result.summary += f"  Тема {idx}: {', '.join(words)}\n"

        result.summary += "\n  Распределение тем:\n"
        for topic_name, percentage in topic_percentages.items():
            result.summary += f"    {topic_name}: {percentage:.2f}%\n"

        return result

    def find_optimal_number_of_topics(
            self,
            blocks: List[str],
            min_topics: Optional[int] = None,
            max_topics: Optional[int] = None,
            n_top_words: Optional[int] = None,
    ) -> Tuple[int, Dict[int, List[str]], List[float]]:
        """
        Находит оптимальное количество тем для набора текстовых блоков, используя модель LDA.

        Args:
            blocks: Список текстовых блоков для анализа
            min_topics: Минимальное количество тем для рассмотрения
            max_topics: Максимальное количество тем для рассмотрения
            n_top_words: Количество топ-слов для каждой темы

        Returns:
            Кортеж содержащий:
            - Оптимальное количество тем
            - Словарь с топ-словами для каждой темы
            - Список перплексий для каждого количества тем
        """
        if min_topics is None:
            min_topics = self.config.analyse.min_topics
        if max_topics is None:
            max_topics = self.config.analyse.max_topics
        if n_top_words is None:
            n_top_words = self.config.topic.top_words_count

        # Используем настройку для векторизатора
        if not hasattr(self, "vectorizer") or self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                stop_words=get_stop_words(),
                max_df=self.config.topic.max_df_vectorizer,
            )

        X = self.vectorizer.fit_transform(blocks)

        perplexities: List[float] = []
        topic_models: Dict[int, LatentDirichletAllocation] = {}
        topic_keywords: Dict[int, List[List[str]]] = {}

        for n_topics in range(min_topics, max_topics + 1):
            lda = LatentDirichletAllocation(
                n_components=n_topics, random_state=42
            )
            lda.fit(X)

            perp: float = lda.perplexity(X)
            perplexities.append(perp)
            topic_models[n_topics] = lda

            keywords: List[List[str]] = []
            feature_names = self.vectorizer.get_feature_names_out()

            for topic in lda.components_:
                top_features: List[str] = [
                    feature_names[i]
                    for i in topic.argsort()[: -n_top_words - 1: -1]
                ]
                keywords.append(top_features)

            topic_keywords[n_topics] = keywords

        optimal_n: int = min(
            range(min_topics, max_topics + 1),
            key=lambda n: topic_models[n].perplexity(X),
        )

        topic_keywords_flat: Dict[int, List[str]] = {
            idx + 1: words
            for idx, words in enumerate(topic_keywords[optimal_n])
        }

        return optimal_n, topic_keywords_flat, perplexities

    def perform_topic_modeling(
            self, blocks: List[str], num_topics: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Выполняет тематическое моделирование на основе заданных текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа
            num_topics: Количество тем для моделирования

        Returns:
            Словарь с процентным распределением тем
        """
        if num_topics is None:
            num_topics = self.config.topic.num_topics_defaults

        X = self.vectorizer.fit_transform(blocks)

        lda = LatentDirichletAllocation(
            n_components=num_topics, random_state=42
        )
        lda.fit(X)

        topic_distributions: np.ndarray = lda.transform(X)
        topic_sum: np.ndarray = topic_distributions.sum(axis=0)

        total: float = topic_sum.sum()
        percentages: Dict[str, float] = {
            f"Тема {i + 1}": (topic_sum[i] / total * 100)
            for i in range(num_topics)
        }

        return percentages

    def save_optimal_topics_plots(
            self,
            perplexities: List[float],
            min_topics: int,
            max_topics: int,
            topic_percentages: Dict[str, float],
            topic_keywords: Dict[int, List[str]],
    ) -> None:
        """
        Сохраняет графики оптимального количества тем и процентного распределения тем.

        Args:
            perplexities: Список значений перплексии для каждого количества тем
            min_topics: Минимальное количество тем
            max_topics: Максимальное количество тем
            topic_percentages: Словарь с процентным распределением тем
            topic_keywords: Словарь с ключевыми словами для каждой темы
        """
        topic_range: List[int] = list(range(min_topics, max_topics + 1))

        plt.figure(figsize=(8, 6))
        plt.plot(topic_range, perplexities, marker="o")
        plt.title("Перплексия LDA модели при разном числе тем")
        plt.xlabel("Число тем")
        plt.ylabel("Перплексия")
        plt.grid(True)
        plt.savefig(
            self.save_figure("optimal_topics_perplexity.png"),
            bbox_inches="tight",
        )
        plt.close()

        labels: List[str] = [
            f"Тема {i}: " + ", ".join(topic_keywords[i][:3])
            for i in topic_keywords
        ]
        percentages: List[float] = list(topic_percentages.values())

        plt.figure(figsize=(10, 6))
        x: np.ndarray = np.arange(len(labels))
        plt.bar(x, percentages, edgecolor="black")
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=9)
        plt.title("Процентное распределение тем в тексте")
        plt.xlabel("Темы")
        plt.ylabel("Процент")
        plt.tight_layout()
        plt.savefig(
            self.save_figure("topic_distribution_bar.png"),
            bbox_inches="tight",
        )
        plt.close()
