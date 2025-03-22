"""
Модуль для анализа повторяемости и стиля текста.
"""

import os
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.text_processor import TextProcessor


class RepetitionAnalyser(BaseAnalyser):
    """
    Класс для анализа повторяемости и стиля текста.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует анализатор повторяемости.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.text_processor = TextProcessor()

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет анализ повторяемости и стиля текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Объединяем все блоки в один текст для общего анализа
        overall_text = " ".join(blocks)

        # Анализируем повторяемость и стиль
        repetition_metrics = self.analyze_repetition_and_style(overall_text)

        # Сохраняем график топ-слов
        if repetition_metrics["top_words"]:
            self.save_top_words_chart(repetition_metrics["top_words"])

        # Заполняем метрики
        result.metrics.update(
            {
                "repetition_ratio": repetition_metrics["repetition_ratio"],
                "top_words": repetition_metrics["top_words"],
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "top_words_chart": os.path.join(
                    self.config.output.output_dir, "top_words.png"
                )
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Метрики повторяемости и стиля:\n"
        result.summary += f"  Коэффициент повторяемости: {repetition_metrics['repetition_ratio']:.3f}\n"
        result.summary += "  Топ 10 наиболее частотных слов:\n"

        for word, freq in repetition_metrics["top_words"][:10]:
            result.summary += f"    {word}: {freq}\n"

        # Добавляем интерпретацию
        if (
                repetition_metrics["repetition_ratio"]
                > self.config.repetition.high_repetition_threshold
        ):
            result.summary += "\n  Высокая повторяемость слов, текст может восприниматься монотонно.\n"
        elif (
                repetition_metrics["repetition_ratio"]
                > self.config.repetition.medium_repetition_threshold
        ):
            result.summary += "\n  Средняя повторяемость слов, типичная для большинства текстов.\n"
        else:
            result.summary += "\n  Низкая повторяемость слов, текст лексически разнообразен.\n"

        return result

    def analyze_repetition_and_style(self, text: str) -> Dict[str, Any]:
        """
        Анализирует повторяемость и стиль переданного текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с результатами анализа, содержащий:
            - "top_words": список кортежей (слово, частота)
            - "repetition_ratio": отношение повторяющихся слов к общему числу слов
        """
        words: List[str] = [
            w.lower()
            for w in self.text_processor.tokenize(text)
            if w.isalpha()
        ]

        counter: Counter = Counter(words)
        repeated_words_count: int = sum(
            count for count in counter.values() if count > 1
        )

        most_repeatable: List[Tuple[str, int]] = counter.most_common(
            self.config.repetition.top_words_limit
        )
        repetition_ratio: float = (
            repeated_words_count / len(words) if words else 0
        )

        return {
            "top_words": most_repeatable,
            "repetition_ratio": repetition_ratio,
        }

    def save_top_words_chart(self, top_words: List[Tuple[str, int]]) -> None:
        """
        Сохраняет диаграмму топ-слов.

        Args:
            top_words: Список кортежей (слово, частота)
        """
        words, freqs = zip(
            *top_words[: self.config.repetition.top_chart_words]
        )  # Используем вложенную структуру

        plt.figure(figsize=(10, 6))
        plt.bar(words, freqs, edgecolor="black")
        plt.title("Топ 10 наиболее частотных слов")
        plt.xlabel("Слова")
        plt.ylabel("Частота")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.save_figure("top_words.png"))
        plt.close()
