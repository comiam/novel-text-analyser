"""
API модуль для анализа персонажей и их диалогов.
"""

from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

from novel_analyser.analysers.character import (
    CharacterAnalyser as CharacterAnalyserCore,
)
from novel_analyser.core.base_analyser import AnalysisResult
from novel_analyser.core.config import get_config, configure
from novel_analyser.core.plugins import create_parser


class CharacterMetrics(BaseModel):
    """Модель для хранения метрик персонажа."""

    replica_count: int = Field(default=0, description="Количество реплик")
    total_duration: float = Field(
        default=0.0, description="Общая длительность диалогов"
    )
    avg_replica_duration: float = Field(
        default=0.0, description="Средняя длительность реплики"
    )
    replica_frequency: float = Field(default=0.0, description="Частота реплик")
    sentiment: float = Field(default=0.0, description="Эмоциональная окраска")
    num_replicas: int = Field(default=0, description="Количество реплик")


class CharacterAnalyser:
    """
    API класс для анализа персонажей и их диалогов.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует анализатор персонажей.

        Args:
            config: Словарь с параметрами конфигурации.
                   Если None, используется конфигурация по умолчанию.
        """
        if config:
            configure(config)
        self.config = get_config()
        self.analyser = CharacterAnalyserCore(self.config)

    def analyse_text(self, text: str) -> AnalysisResult:
        """
        Анализирует персонажей и их диалоги в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Результат анализа
        """
        parser = create_parser()
        blocks = parser.parse_blocks(text, raw_style=True)
        return self.analyser.analyse(blocks)

    def analyse_file(self, file_path: str) -> AnalysisResult:
        """
        Анализирует персонажей и их диалоги в тексте из файла.

        Args:
            file_path: Путь к файлу с текстом

        Returns:
            Результат анализа
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.analyse_text(content)

    def extract_dialogues(self, text: str) -> Dict[str, List[str]]:
        """
        Извлекает диалоги персонажей из текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь, где ключи - имена персонажей, а значения - списки их реплик
        """
        # Создаем парсер для текста
        parser = create_parser()

        # Парсим текст на блоки и извлекаем диалоги
        blocks = parser.parse_blocks(text, raw_style=True)
        return parser.parse_character_dialogues(blocks)

    def get_character_sentiments(self, text: str) -> Dict[str, float]:
        """
        Определяет эмоциональную окраску реплик каждого персонажа.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с именами персонажей и их средней эмоциональной окраской
        """
        parser = create_parser()
        blocks = parser.parse_blocks(text, raw_style=True)
        character_dialogues = parser.parse_character_dialogues(blocks)
        return self.analyser.analyze_character_sentiments(character_dialogues)

    def get_character_metrics(self, text: str) -> Dict[str, CharacterMetrics]:
        """
        Получает полный набор метрик для всех персонажей.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с именами персонажей и объектами их метрик
        """
        parser = create_parser()
        blocks = parser.parse_blocks(text, raw_style=True)
        character_dialogues = parser.parse_character_dialogues(blocks)

        # Вычисляем метрики
        replica_counts, durations, avg_durations, frequencies = (
            self.analyser.compute_character_metrics(
                character_dialogues, len(blocks)
            )
        )

        # Вычисляем настроения
        sentiments = self.analyser.analyze_character_sentiments(
            character_dialogues
        )

        # Формируем результат
        result = {}
        for char in character_dialogues.keys():
            result[char] = CharacterMetrics(
                replica_count=replica_counts.get(char, 0),
                total_duration=durations.get(char, 0.0),
                avg_replica_duration=avg_durations.get(char, 0.0),
                replica_frequency=frequencies.get(char, 0.0),
                sentiment=sentiments.get(char, 0.0),
                num_replicas=len(character_dialogues.get(char, [])),
            )

        return result
