"""
API модуль для анализа эмоциональной окраски текста.
"""

from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field

from novel_analyser.analysers.sentiment import (
    SentimentAnalyser as SentimentAnalyserCore,
)
from novel_analyser.core.base_analyser import AnalysisResult
from novel_analyser.core.config import get_config, configure
from novel_analyser.core.parser import parse_blocks


class SentimentAnalysisConfig(BaseModel):
    """Конфигурация для анализа настроений."""

    model_name: str = Field(
        default="cointegrated/rubert-tiny-sentiment-balanced",
        description="Название модели для анализа настроений",
    )
    device: Optional[str] = Field(
        default=None, description="Устройство для вычислений"
    )
    device_id: int = Field(default=0, description="ID устройства GPU")
    positive_threshold: float = Field(
        default=0.05, description="Порог для положительного настроения"
    )
    negative_threshold: float = Field(
        default=-0.05, description="Порог для отрицательного настроения"
    )


class SentimentAnalyser:
    """
    API класс для анализа эмоциональной окраски текста.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует анализатор эмоциональной окраски.

        Args:
            config: Словарь с параметрами конфигурации.
                   Если None, используется конфигурация по умолчанию.
        """
        if config:
            configure(config)
        self.config = get_config()
        self.analyser = SentimentAnalyserCore(self.config)

    def analyse_text(
            self,
            text: str,
            weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> AnalysisResult:
        """
        Анализирует эмоциональную окраску текста.

        Args:
            text: Текст для анализа
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Результат анализа
        """
        blocks = parse_blocks(text)
        return self.analyser.analyse(blocks, weighting_strategy)

    def analyse_file(
            self,
            file_path: str,
            weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> AnalysisResult:
        """
        Анализирует эмоциональную окраску текста из файла.

        Args:
            file_path: Путь к файлу с текстом
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Результат анализа
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.analyse_text(content, weighting_strategy)

    def analyse_texts(
            self,
            texts: List[str],
            weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> List[float]:
        """
        Анализирует эмоциональную окраску списка текстов.

        Args:
            texts: Список текстов для анализа
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Список значений эмоциональной окраски для каждого текста
        """
        return [
            self.analyser.analyze_long_text(text, weighting_strategy)
            for text in texts
        ]
