"""
Модуль для анализа эмоциональной окраски текста.
"""

import os
from typing import List, Optional, Literal

import numpy as np

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig, get_config
from novel_analyser.core.interfaces.sentiment import BaseSentimentProcessor
from novel_analyser.utils.plot import (
    save_sentiment_histogram,
    save_sentiment_pie_chart,
)
from novel_analyser.utils.stat import filter_outliers_by_percentiles


class SentimentAnalyser(BaseAnalyser):
    """
    Класс для анализа эмоциональной окраски текста с использованием трансформеров.
    """

    def __init__(
        self,
        sentiment_processor: BaseSentimentProcessor,
        config: Optional[AnalyserConfig] = None,
    ):
        """
        Инициализирует анализатор настроений.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)

        # Создаем обработчик эмоциональной окраски
        self.processor = sentiment_processor
        # Get sentiment config from the main config
        self.sentiment_analyze_config = get_config().sentiment_analyze

    def analyse(
        self,
        blocks: List[str],
        weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> AnalysisResult:
        """
        Выполняет анализ эмоциональной окраски текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Получаем оценки настроений для всех блоков
        sentiments = [
            self.processor.analyze_long_text(block, weighting_strategy)
            for block in blocks
        ]

        # Фильтруем выбросы по перцентилям из конфигурации
        filtered_sentiments = filter_outliers_by_percentiles(
            sentiments,
            self.sentiment_analyze_config.sentiment_percentile_lower,
            self.sentiment_analyze_config.sentiment_percentile_upper,
        )

        # Получаем пороги из конфигурации
        positive_threshold = self.sentiment_analyze_config.positive_threshold
        negative_threshold = self.sentiment_analyze_config.negative_threshold

        # Вычисляем основные метрики, используя отфильтрованные данные для средних значений
        avg_sentiment = (
            float(np.mean(filtered_sentiments)) if filtered_sentiments else 0.0
        )
        pos_blocks = sum(1 for s in sentiments if s > positive_threshold)
        neg_blocks = sum(1 for s in sentiments if s < negative_threshold)
        neu_blocks = len(sentiments) - pos_blocks - neg_blocks

        pos_ratio = pos_blocks / len(sentiments) if sentiments else 0
        neg_ratio = neg_blocks / len(sentiments) if sentiments else 0
        neu_ratio = neu_blocks / len(sentiments) if sentiments else 0

        # Сохраняем гистограмму и круговую диаграмму
        save_sentiment_histogram(
            sentiments, self.save_figure("sentiment_histogram.png")
        )

        save_sentiment_pie_chart(
            sentiments,
            self.save_figure("sentiment_pie_chart.png"),
            positive_threshold,
            negative_threshold,
        )

        # Заполняем метрики
        result.metrics.update(
            {
                "avg_sentiment": avg_sentiment,
                "pos_blocks": pos_blocks,
                "neg_blocks": neg_blocks,
                "neu_blocks": neu_blocks,
                "pos_ratio": pos_ratio,
                "neg_ratio": neg_ratio,
                "neu_ratio": neu_ratio,
                "sentiments": sentiments,
                "filtered_sentiments": filtered_sentiments,
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "sentiment_histogram": os.path.join(
                    self.config.output.output_dir, "sentiment_histogram.png"
                ),
                "sentiment_pie_chart": os.path.join(
                    self.config.output.output_dir, "sentiment_pie_chart.png"
                ),
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Анализ эмоциональной окраски:\n"
        result.summary += (
            f"  Средний эмоциональный окрас: {avg_sentiment:.3f}\n"
        )
        result.summary += (
            f"  Положительные блоки: {pos_blocks} ({pos_ratio * 100:.1f}%)\n"
        )
        result.summary += (
            f"  Нейтральные блоки: {neu_blocks} ({neu_ratio * 100:.1f}%)\n"
        )
        result.summary += (
            f"  Отрицательные блоки: {neg_blocks} ({neg_ratio * 100:.1f}%)\n"
        )

        # Добавляем интерпретацию
        if avg_sentiment > 0.2:
            result.summary += (
                "  Общая эмоциональная окраска текста: выраженно позитивная\n"
            )
        elif avg_sentiment > 0.05:
            result.summary += (
                "  Общая эмоциональная окраска текста: умеренно позитивная\n"
            )
        elif avg_sentiment > -0.05:
            result.summary += (
                "  Общая эмоциональная окраска текста: нейтральная\n"
            )
        elif avg_sentiment > -0.2:
            result.summary += (
                "  Общая эмоциональная окраска текста: умеренно негативная\n"
            )
        else:
            result.summary += (
                "  Общая эмоциональная окраска текста: выраженно негативная\n"
            )

        return result

    def analyze_long_text(
        self,
        text: str,
        weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> float:
        """
        Анализирует длинный текст, разбивая его на фрагменты и объединяя результаты.

        Args:
            text: Текст для анализа
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Итоговая оценка настроения
        """
        return self.processor.analyze_long_text(text, weighting_strategy)
