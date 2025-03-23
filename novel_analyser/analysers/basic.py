"""
Модуль базового анализа текста.
"""

import os
import re
from typing import List, Optional

import numpy as np
from razdel import tokenize

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.utils.plot import save_histogram
from novel_analyser.utils.stat import filter_outliers_by_percentiles


class BasicAnalyser(BaseAnalyser):
    """
    Класс для базового анализа текста.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует базовый анализатор.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)

    def analyse(
        self, blocks: List[str], raw_blocks: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Выполняет базовый анализ текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа
            raw_blocks: Список сырых текстовых блоков (используется для анализа условий if-else)

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Вычисляем время чтения для каждого блока
        reading_times = [self.compute_reading_time(block) for block in blocks]

        # Если есть сырые блоки, анализируем условия if-else
        if raw_blocks:
            if_counts = [
                self.count_if_conditions(block) for block in raw_blocks
            ]
            if_else_counts = [
                self.count_if_else_conditions(block) for block in raw_blocks
            ]
        else:
            if_counts = []
            if_else_counts = []

        # Фильтруем выбросы
        filtered_reading_times = filter_outliers_by_percentiles(reading_times)

        # Вычисляем метрики
        total_if = sum(if_counts)
        total_if_else = sum(if_else_counts)
        avg_reading_time = (
            np.mean(filtered_reading_times) if filtered_reading_times else 0
        )

        # Сохраняем гистограммы
        if filtered_reading_times:
            save_histogram(
                filtered_reading_times,
                "Гистограмма времени чтения блоков",
                "Время чтения (секунды)",
                "Количество блоков",
                self.save_figure(
                    "reading_time_histogram.png"
                ),  # Use self.save_figure instead of self.config.save_figure
            )

        non_zero_if = [c for c in if_counts if c > 0]
        if non_zero_if:
            save_histogram(
                non_zero_if,
                "Гистограмма количества if условий в блоках",
                "Количество if",
                "Количество блоков",
                self.save_figure(
                    "if_conditions_histogram.png"
                ),  # Use self.save_figure instead of self.config.save_figure
            )

        non_zero_if_else = [c for c in if_else_counts if c > 0]
        if non_zero_if_else:
            save_histogram(
                non_zero_if_else,
                "Гистограмма количества if_else условий в блоках",
                "Количество if_else",
                "Количество блоков",
                self.save_figure(
                    "if_else_conditions_histogram.png"
                ),  # Use self.save_figure instead of self.config.save_figure
            )

        # Заполняем метрики
        result.metrics.update(
            {
                "total_blocks": len(blocks),
                "total_if_conditions": total_if,
                "total_if_else_conditions": total_if_else,
                "avg_reading_time": avg_reading_time,
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "reading_time_histogram": os.path.join(
                    self.config.output.output_dir, "reading_time_histogram.png"
                ),
                "if_conditions_histogram": os.path.join(
                    self.config.output.output_dir,
                    "if_conditions_histogram.png",
                ),
                "if_else_conditions_histogram": os.path.join(
                    self.config.output.output_dir,
                    "if_else_conditions_histogram.png",
                ),
            }
        )

        # Заполняем текстовое резюме
        result.summary = f"Базовый анализ:\n"
        result.summary += f"  Количество блоков: {len(blocks)}\n"
        result.summary += f"  Количество if-условий в блоках: {total_if}\n"
        result.summary += (
            f"  Количество if-else-условий в блоках: {total_if_else}\n"
        )
        result.summary += (
            f"  Среднее время чтения блока: {avg_reading_time:.2f} сек\n"
        )

        return result

    def compute_reading_time(self, block: str, wpm: float = 150.0) -> float:
        """
        Вычисляет время чтения блока в секундах.

        Args:
            block: Текст блока
            wpm: Скорость чтения (слов в минуту)

        Returns:
            Время чтения в секундах
        """
        if wpm is None:
            wpm = self.config.analyse.reading_speed_wpm

        words: List[str] = [token.text for token in tokenize(block)]
        words_per_sec: float = wpm / 60
        return len(words) / words_per_sec if words else 0.0

    @staticmethod
    def count_if_conditions(block: str) -> int:
        """
        Считает количество if-условий в блоке.

        Args:
            block: Текст блока

        Returns:
            Число вхождений if
        """
        return len(re.findall(r"if", block))

    @staticmethod
    def count_if_else_conditions(block: str) -> int:
        """
        Считает количество ветвлений if-else в блоке.

        Args:
            block: Текст блока

        Returns:
            Число вхождений if else
        """
        return len(re.findall(r"(?s)if.*?else", block, re.IGNORECASE))
