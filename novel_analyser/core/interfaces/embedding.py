"""
Интерфейсы и базовые классы для обработчиков эмбеддингов текста.

Данный модуль определяет протоколы и абстрактные классы,
которые должны быть реализованы всеми обработчиками эмбеддингов в системе.
"""

import abc
from typing import List, Protocol

import numpy as np


class EmbeddingProcessorProtocol(Protocol):
    """
    Протокол для обработчиков эмбеддингов текста.

    Определяет обязательные методы, которые должен реализовать обработчик эмбеддингов.
    """

    def encode(
        self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги.

        Args:
            texts: Список текстов для кодирования
            show_progress: Показывать ли прогресс кодирования

        Returns:
            Массив эмбеддингов размерности (len(texts), embedding_dim)
        """
        ...


class BaseEmbeddingEncoder(abc.ABC):
    """
    Базовый абстрактный класс для всех обработчиков эмбеддингов текста.

    Предоставляет абстрактные методы, которые должны быть реализованы в наследниках,
    а также некоторые общие методы, которые могут быть переопределены при необходимости.
    """

    def __init__(self):
        """Инициализирует обработчик эмбеддингов."""
        pass

    @abc.abstractmethod
    def encode(
        self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги.

        Args:
            texts: Список текстов для кодирования
            show_progress: Показывать ли прогресс кодирования

        Returns:
            Массив эмбеддингов размерности (len(texts), embedding_dim)
        """
        pass

    @abc.abstractmethod
    def clear_cache(self) -> None:
        """Очищает кэш эмбеддингов."""
        pass

    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.

        Returns:
            Размерность эмбеддинга
        """
        # По умолчанию определяем размерность по одному тестовому эмбеддингу
        test_embedding = self.encode(["Тестовый текст"], show_progress=False)
        return test_embedding.shape[1]
