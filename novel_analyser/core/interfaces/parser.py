"""
Интерфейсы и базовые классы для парсеров текста.

Данный модуль определяет протоколы и абстрактные классы,
которые должны быть реализованы всеми парсерами текста в системе.
"""

import abc
from typing import Dict, List, Protocol


class ParserProtocol(Protocol):
    """
    Протокол для парсеров текста.

    Определяет обязательные методы, которые должен реализовать парсер текста.
    """

    def parse_blocks(self, text: str, raw_style: bool = False) -> List[str]:
        """
        Разбирает текст на блоки.

        Args:
            text: Исходный текст для разбора
            raw_style: Флаг для сохранения блоков в сыром виде

        Returns:
            Список текстовых блоков
        """
        ...

    def parse_character_dialogues(
        self, blocks: List[str]
    ) -> Dict[str, List[str]]:
        """
        Извлекает диалоги персонажей из блоков текста.

        Args:
            blocks: Список текстовых блоков

        Returns:
            Словарь с именами персонажей и их репликами
        """
        ...

    def extract_sentences(self, text: str) -> List[str]:
        """
        Извлекает предложения из текста.

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        ...

    def extract_all_sentences(self, blocks: List[str]) -> List[str]:
        """
        Извлекает все предложения из списка текстовых блоков.

        Args:
            blocks: Список текстовых блоков

        Returns:
            Список всех предложений из всех блоков
        """
        ...


class BaseParser(abc.ABC):
    """
    Базовый абстрактный класс для всех парсеров текста.

    Предоставляет абстрактные методы, которые должны быть реализованы в наследниках,
    а также некоторые общие методы, которые могут быть переопределены при необходимости.
    """

    def __init__(self):
        """Инициализирует парсер."""
        pass

    @abc.abstractmethod
    def parse_blocks(self, text: str, raw_style: bool = False) -> List[str]:
        """
        Разбирает текст на блоки.

        Args:
            text: Исходный текст для разбора
            raw_style: Флаг для сохранения блоков в сыром виде

        Returns:
            Список текстовых блоков
        """
        pass

    @abc.abstractmethod
    def parse_character_dialogues(
        self, blocks: List[str]
    ) -> Dict[str, List[str]]:
        """
        Извлекает диалоги персонажей из блоков текста.

        Args:
            blocks: Список текстовых блоков

        Returns:
            Словарь с именами персонажей и их репликами
        """
        pass

    @abc.abstractmethod
    def extract_sentences(self, text: str) -> List[str]:
        """
        Извлекает предложения из текста.

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        pass

    def extract_all_sentences(self, blocks: List[str]) -> List[str]:
        """
        Извлекает все предложения из списка текстовых блоков.

        Args:
            blocks: Список текстовых блоков

        Returns:
            Список всех предложений из всех блоков
        """
        sentences: List[str] = []
        for block in blocks:
            sents = self.extract_sentences(block)
            sentences.extend(sents)
        return sentences
