"""
Базовые классы и интерфейсы для всех анализаторов.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field

from novel_analyser.core.config import AnalyserConfig, get_config


class AnalysisResult(BaseModel):
    """
    Класс для хранения результатов анализа текста.
    """

    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Словарь с метриками анализа"
    )
    figures: Dict[str, str] = Field(
        default_factory=dict,
        description="Словарь с путями к сохраненным изображениям",
    )
    summary: str = Field(
        default="", description="Текстовое резюме результатов анализа"
    )

    def extend(self, other: "AnalysisResult") -> "AnalysisResult":
        """
        Расширяет текущий результат данными из другого результата.

        Args:
            other: Другой результат анализа

        Returns:
            AnalysisResult: Обновленный объект (self)
        """
        self.metrics.update(other.metrics)
        self.figures.update(other.figures)
        if other.summary:
            if self.summary:
                self.summary += "\n\n" + other.summary
            else:
                self.summary = other.summary
        return self

    def save_metrics(self, filepath: str) -> None:
        """
        Сохраняет метрики и резюме в текстовый файл.

        Args:
            filepath: Путь к файлу для сохранения
        """
        with open(filepath, "w", encoding="utf-8") as f:
            if self.summary:
                f.write(self.summary + "\n\n")

            f.write("Детальные метрики:\n")
            for name, value in self.metrics.items():
                f.write(f"{name}: {value}\n")


class BaseAnalyser(ABC):
    """
    Абстрактный базовый класс для всех анализаторов текста.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует анализатор.

        Args:
            config: Конфигурация анализатора, если None, то используется глобальная конфигурация
        """
        self.config = config or get_config()

    @abstractmethod
    def analyse(self, *args, **kwargs) -> AnalysisResult:
        """
        Выполняет анализ и возвращает результат.

        Returns:
            AnalysisResult: Результат анализа
        """
        pass

    def save_figure(
        self, figure_name: str, subdirectory: Optional[str] = None
    ) -> str:
        """
        Создает полный путь для сохранения изображения с учетом конфигурации.

        Args:
            figure_name: Имя файла для сохранения
            subdirectory: Поддиректория в директории вывода

        Returns:
            str: Полный путь к файлу
        """
        base_dir = self.config.output.output_dir
        if subdirectory:
            dir_path = os.path.join(base_dir, subdirectory)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            return os.path.join(dir_path, figure_name)
        return os.path.join(base_dir, figure_name)
