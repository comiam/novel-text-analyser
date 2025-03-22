"""
Модуль для анализа нарративной структуры текста.
"""

import os
from typing import Dict, List, Optional

import numpy as np

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.text_processor import TextProcessor
from novel_analyser.utils.plot import save_histogram
from novel_analyser.utils.stat import filter_outliers_by_percentiles


class NarrativeAnalyser(BaseAnalyser):
    """
    Класс для анализа нарративной структуры и ритма повествования.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует анализатор нарратива.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.text_processor = TextProcessor()

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет анализ нарративной структуры текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Вычисляем метрики нарративного ритма для каждого блока
        narrative_stats = [
            self.analyze_narrative_rhythm(block) for block in blocks
        ]

        # Извлекаем данные для гистограмм
        sentence_counts = [stat["num_sentences"] for stat in narrative_stats]
        avg_words_list = [
            stat["avg_words_per_sentence"] for stat in narrative_stats
        ]

        # Фильтруем выбросы для среднего числа слов в предложении
        filtered_avg_words = filter_outliers_by_percentiles(avg_words_list)

        # Вычисляем общие метрики
        overall_avg_sentences = (
            np.mean(sentence_counts) if sentence_counts else 0
        )
        overall_avg_words = (
            np.mean(filtered_avg_words) if filtered_avg_words else 0
        )

        # Сохраняем гистограммы
        save_histogram(
            sentence_counts,
            "Гистограмма числа предложений в блоках",
            "Число предложений",
            "Количество блоков",
            self.save_figure("narrative_sentence_count_hist.png"),
        )

        save_histogram(
            filtered_avg_words,
            "Гистограмма среднего числа слов в предложении",
            "Среднее число слов в предложении",
            "Количество блоков",
            self.save_figure("narrative_avg_words_hist.png"),
        )

        # Заполняем метрики
        result.metrics.update(
            {
                "avg_sentences_per_block": overall_avg_sentences,
                "avg_words_per_sentence": overall_avg_words,
                "total_sentences": sum(
                    stat["num_sentences"] for stat in narrative_stats
                ),
                "sentence_counts_variability": (
                    np.std(sentence_counts) if sentence_counts else 0
                ),
                "avg_words_variability": (
                    np.std(filtered_avg_words) if filtered_avg_words else 0
                ),
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "sentence_count_histogram": os.path.join(
                    self.config.output.output_dir,
                    "narrative_sentence_count_hist.png",
                ),
                "avg_words_histogram": os.path.join(
                    self.config.output.output_dir,
                    "narrative_avg_words_hist.png",
                ),
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Анализ повествовательного ритма:\n"
        result.summary += f"  Среднее число предложений в блоке: {overall_avg_sentences:.2f}\n"
        result.summary += (
            f"  Среднее число слов в предложении: {overall_avg_words:.2f}\n"
        )
        result.summary += f"  Вариативность числа предложений: {result.metrics['sentence_counts_variability']:.2f}\n"
        result.summary += f"  Вариативность средней длины предложений: {result.metrics['avg_words_variability']:.2f}\n"

        # Оценка повествовательного ритма
        if (
                result.metrics["sentence_counts_variability"]
                > self.config.narrative.dynamic_rhythm_threshold
                and result.metrics["avg_words_variability"]
                > self.config.narrative.dynamic_rhythm_threshold
        ):
            result.summary += "  Ритм повествования: динамичный, с значительными вариациями\n"
        elif (
                result.metrics["sentence_counts_variability"]
                > self.config.narrative.moderate_rhythm_threshold
                or result.metrics["avg_words_variability"]
                > self.config.narrative.moderate_rhythm_threshold
        ):
            result.summary += "  Ритм повествования: умеренно вариативный\n"
        else:
            result.summary += (
                "  Ритм повествования: ровный, с малой вариативностью\n"
            )

        return result

    def analyze_narrative_rhythm(self, text: str) -> Dict[str, float]:
        """
        Анализирует ритм повествования в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с результатами анализа, содержащий количество предложений и среднее количество слов на предложение
        """
        sentences: List[str] = self.text_processor.extract_sentences(text)
        num_sentences: int = len(sentences)

        total_words: int = sum(
            len(self.text_processor.tokenize(sentence))
            for sentence in sentences
        )
        avg_words: float = (
            total_words / num_sentences if num_sentences > 0 else 0
        )

        return {
            "num_sentences": num_sentences,
            "avg_words_per_sentence": avg_words,
        }
