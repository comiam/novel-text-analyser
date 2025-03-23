"""
Модуль для анализа персонажей и их диалогов.
"""

import os
from typing import Dict, List, Tuple, Optional

import numpy as np

from novel_analyser.analysers.basic import BasicAnalyser
from novel_analyser.analysers.sentiment import SentimentAnalyser
from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.interfaces.parser import BaseParser
from novel_analyser.utils.plot import save_bar_chart


class CharacterAnalyser(BaseAnalyser):
    """
    Класс для анализа персонажей и их диалогов.
    """

    def __init__(
        self,
        text_parser: BaseParser,
        sentiment_analyser: SentimentAnalyser,
        basic_analyser: BasicAnalyser,
        config: Optional[AnalyserConfig] = None,
    ):
        """
        Инициализирует анализатор персонажей.

        Args:
            text_parser: Парсер для извлечения диалогов из файла
            sentiment_analyser: Анализатор настроений
            basic_analyser: Базовый анализатор
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.sentiment_analyser = sentiment_analyser
        self.basic_analyser = basic_analyser
        self.text_parser = text_parser

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет анализ персонажей и их диалогов в текстовых блоках.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Извлекаем диалоги персонажей
        character_dialogues = self.text_parser.parse_character_dialogues(
            blocks
        )

        if not character_dialogues:
            result.summary = "Анализ персонажей:\n  В тексте не обнаружено диалогов персонажей.\n"
            return result

        # Вычисляем метрики для персонажей
        replica_counts, durations, avg_durations, frequencies = (
            self.compute_character_metrics(character_dialogues, len(blocks))
        )

        # Анализируем настроения персонажей
        character_sentiments = self.analyze_character_sentiments(
            character_dialogues
        )

        # Подготавливаем данные для построения диаграмм
        chars = list(character_dialogues.keys())

        # Сохраняем диаграммы
        save_bar_chart(
            chars,
            [replica_counts[char] for char in chars],
            "Количество реплик персонажей",
            "Персонаж",
            "Количество реплик",
            self.save_figure("character_replica_counts.png"),
        )

        save_bar_chart(
            chars,
            [durations[char] for char in chars],
            "Общая длительность диалогов персонажей",
            "Персонаж",
            "Длительность (сек)",
            self.save_figure("character_dialogue_durations.png"),
        )

        save_bar_chart(
            chars,
            [avg_durations[char] for char in chars],
            "Средняя длительность реплики персонажа",
            "Персонаж",
            "Средняя длительность (сек)",
            self.save_figure("character_avg_durations.png"),
        )

        save_bar_chart(
            chars,
            [frequencies[char] for char in chars],
            "Частота реплик персонажей на блок",
            "Персонаж",
            "Частота",
            self.save_figure("character_replica_frequencies.png"),
        )

        save_bar_chart(
            chars,
            [character_sentiments.get(char, 0) for char in chars],
            "Среднее настроение персонажей",
            "Персонаж",
            "Настроение (-1: негативное, 1: позитивное)",
            self.save_figure("character_sentiments.png"),
        )

        # Заполняем метрики
        result.metrics.update(
            {
                "num_characters": len(character_dialogues),
                "total_replicas": sum(replica_counts.values()),
                "total_dialogue_duration": sum(durations.values()),
                "character_replica_counts": replica_counts,
                "character_durations": durations,
                "character_avg_durations": avg_durations,
                "character_frequencies": frequencies,
                "character_sentiments": character_sentiments,
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "character_replica_counts": os.path.join(
                    self.config.output.output_dir,
                    "character_replica_counts.png",
                ),
                "character_dialogue_durations": os.path.join(
                    self.config.output.output_dir,
                    "character_dialogue_durations.png",
                ),
                "character_avg_durations": os.path.join(
                    self.config.output.output_dir,
                    "character_avg_durations.png",
                ),
                "character_replica_frequencies": os.path.join(
                    self.config.output.output_dir,
                    "character_replica_frequencies.png",
                ),
                "character_sentiments": os.path.join(
                    self.config.output.output_dir, "character_sentiments.png"
                ),
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Анализ персонажей:\n"
        result.summary += (
            f"  Всего обнаружено персонажей: {len(character_dialogues)}\n"
        )
        result.summary += (
            f"  Общее количество реплик: {sum(replica_counts.values())}\n"
        )
        result.summary += f"  Общая длительность диалогов: {sum(durations.values()):.2f} сек\n\n"

        # Детальные метрики по каждому персонажу (топ-5 по количеству реплик)
        top_characters = sorted(
            chars, key=lambda c: replica_counts[c], reverse=True
        )
        for char in top_characters:
            result.summary += f"  Персонаж: {char}\n"
            result.summary += (
                f"    Количество реплик: {replica_counts[char]}\n"
            )
            result.summary += (
                f"    Общая длительность диалогов: {durations[char]:.2f} сек\n"
            )
            result.summary += f"    Средняя длительность реплики: {avg_durations[char]:.2f} сек\n"
            result.summary += (
                f"    Частота реплик на блок: {frequencies[char]:.3f}\n"
            )
            result.summary += f"    Среднее настроение: {character_sentiments.get(char, 0):.3f}\n\n"

        return result

    def compute_character_metrics(
        self, character_dialogues: Dict[str, List[str]], total_blocks: int
    ) -> Tuple[
        Dict[str, int], Dict[str, float], Dict[str, float], Dict[str, float]
    ]:
        """
        Вычисляет метрики для диалогов персонажей.

        Args:
            character_dialogues: Словарь с репликами персонажей
            total_blocks: Общее количество блоков в тексте

        Returns:
            Кортеж из словарей метрик:
            - Количество реплик для каждого персонажа
            - Общая длительность диалогов для каждого персонажа
            - Средняя длительность реплики для каждого персонажа
            - Частота реплик на блок для каждого персонажа
        """
        # Количество реплик персонажей
        character_replica_counts = {
            char: len(replicas)
            for char, replicas in character_dialogues.items()
        }

        # Длительность диалогов персонажей (по времени)
        character_durations = {}
        for char, replicas in character_dialogues.items():
            duration = sum(
                self.basic_analyser.compute_reading_time(replica)
                for replica in replicas
            )
            character_durations[char] = duration

        # Средняя длительность реплики для каждого персонажа
        avg_durations = {
            char: duration / len(replicas) if replicas else 0
            for char, duration, replicas in zip(
                character_dialogues.keys(),
                character_durations.values(),
                character_dialogues.values(),
            )
        }

        # Частота реплик на блок
        replica_frequencies = {
            char: count / total_blocks
            for char, count in character_replica_counts.items()
        }

        return (
            character_replica_counts,
            character_durations,
            avg_durations,
            replica_frequencies,
        )

    def analyze_character_sentiments(
        self, character_dialogues: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """
        Анализирует настроение персонажей на основе их реплик.

        Args:
            character_dialogues: Словарь с репликами персонажей

        Returns:
            Словарь со средним настроением для каждого персонажа
        """
        character_sentiments = {}

        for char, replicas in character_dialogues.items():
            if not replicas:
                continue

            # Анализируем каждую реплику персонажа и вычисляем среднее значение
            sentiments = [
                self.sentiment_analyser.analyze_long_text(
                    replica, weighting_strategy="speech"
                )
                for replica in replicas
            ]

            character_sentiments[char] = float(np.mean(sentiments))

        return character_sentiments
