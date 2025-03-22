"""
API модуль для тематического моделирования текста.
"""

from typing import Dict, List, Optional, Any, Tuple

from pydantic import BaseModel, Field

from novel_analyser.analysers.topic import TopicAnalyser as TopicAnalyserCore
from novel_analyser.core.base_analyser import AnalysisResult
from novel_analyser.core.config import get_config, configure
from novel_analyser.core.parser import parse_blocks


class TopicModelConfig(BaseModel):
    """Конфигурация тематического моделирования."""

    min_topics: int = Field(
        default=2, description="Минимальное количество тем"
    )
    max_topics: int = Field(
        default=20, description="Максимальное количество тем"
    )
    num_topics_default: int = Field(
        default=5, description="Количество тем по умолчанию"
    )
    top_words_count: int = Field(
        default=7, description="Количество топ-слов для темы"
    )


class TopicModeler:
    """
    API класс для тематического моделирования текста.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует анализатор тем.

        Args:
            config: Словарь с параметрами конфигурации.
                   Если None, используется конфигурация по умолчанию.
        """
        if config:
            configure(config)
        self.config = get_config()
        self.analyser = TopicAnalyserCore(self.config)

    def analyse_text(self, text: str) -> AnalysisResult:
        """
        Выполняет тематическое моделирование текста.

        Args:
            text: Текст для анализа

        Returns:
            Результат анализа
        """
        blocks = parse_blocks(text)
        return self.analyser.analyse(blocks)

    def analyse_file(self, file_path: str) -> AnalysisResult:
        """
        Выполняет тематическое моделирование текста из файла.

        Args:
            file_path: Путь к файлу с текстом

        Returns:
            Результат анализа
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.analyse_text(content)

    def extract_topics(
            self, text: str, num_topics: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Извлекает темы из текста и их процентное распределение.

        Args:
            text: Текст для анализа
            num_topics: Количество тем для извлечения.
                       Если None, определяется автоматически.

        Returns:
            Словарь с темами и их процентными долями
        """
        blocks = parse_blocks(text)

        if num_topics is None:
            # Находим оптимальное количество тем
            optimal_n, _, _ = self.analyser.find_optimal_number_of_topics(
                blocks,
                min_topics=self.config.analyse.min_topics,
                max_topics=self.config.analyse.max_topics,
            )
            num_topics = optimal_n

        # Выполняем тематическое моделирование с заданным числом тем
        return self.analyser.perform_topic_modeling(
            blocks, num_topics=num_topics
        )

    def get_optimal_topic_count(
            self, text: str
    ) -> Tuple[int, Dict[int, List[str]]]:
        """
        Определяет оптимальное количество тем для текста и возвращает ключевые слова для каждой темы.

        Args:
            text: Текст для анализа

        Returns:
            Кортеж, содержащий оптимальное количество тем и
            словарь с ключевыми словами для каждой темы.
        """
        blocks = parse_blocks(text)

        optimal_n, topic_keywords, _ = (
            self.analyser.find_optimal_number_of_topics(
                blocks,
                min_topics=self.config.analyse.min_topics,
                max_topics=self.config.analyse.max_topics,
            )
        )

        return optimal_n, topic_keywords
