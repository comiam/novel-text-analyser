"""
Интерфейсы и базовые классы для обработчиков эмоциональной окраски текста.

Данный модуль определяет протоколы и абстрактные классы,
которые должны быть реализованы всеми обработчиками эмоциональной окраски в системе.
"""

import abc
from typing import List, Literal, Protocol


class SentimentProcessorProtocol(Protocol):
    """
    Протокол для обработчиков эмоциональной окраски текста.

    Определяет обязательные методы, которые должен реализовать обработчик эмоциональной окраски.
    """

    def get_sentiment(self, text: str) -> float:
        """
        Вычисляет эмоциональную оценку для текста.

        Args:
            text: Текст для анализа

        Returns:
            Оценка эмоциональной окраски (от -1 до 1)
        """
        ...

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
            Итоговая оценка эмоциональной окраски
        """
        ...

    def split_text_into_chunks(
            self, text: str, max_length: int, overlap: int
    ) -> List[str]:
        """
        Разделяет длинный текст на перекрывающиеся фрагменты.

        Args:
            text: Исходный текст
            max_length: Максимальная длина фрагмента в токенах
            overlap: Количество перекрывающихся токенов между фрагментами

        Returns:
            Список фрагментов текста
        """
        ...


class BaseSentimentProcessor(abc.ABC):
    """
    Базовый абстрактный класс для всех обработчиков эмоциональной окраски текста.

    Предоставляет абстрактные методы, которые должны быть реализованы в наследниках,
    а также некоторые общие методы, которые могут быть переопределены при необходимости.
    """

    def __init__(self):
        """Инициализирует обработчик эмоциональной окраски."""
        pass

    @abc.abstractmethod
    def get_sentiment(self, text: str) -> float:
        """
        Вычисляет эмоциональную оценку для текста.

        Args:
            text: Текст для анализа

        Returns:
            Оценка эмоциональной окраски (от -1 до 1)
        """
        pass

    @abc.abstractmethod
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
            Итоговая оценка эмоциональной окраски
        """
        pass

    @abc.abstractmethod
    def split_text_into_chunks(
            self, text: str, max_length: int, overlap: int
    ) -> List[str]:
        """
        Разделяет длинный текст на перекрывающиеся фрагменты.

        Args:
            text: Исходный текст
            max_length: Максимальная длина фрагмента в токенах
            overlap: Количество перекрывающихся токенов между фрагментами

        Returns:
            Список фрагментов текста
        """
        pass
