"""
Модуль для обработки текста.
"""

import re
from typing import Dict, List

import pymorphy3
from razdel import tokenize


class TextProcessor:
    """
    Класс для обработки и подготовки текста для анализа.
    """

    def __init__(self):
        """
        Инициализирует обработчик текста.
        """
        self.morph = pymorphy3.MorphAnalyzer()

    def tokenize(self, text: str) -> List[str]:
        """
        Разбивает текст на токены.

        Args:
            text: Исходный текст

        Returns:
            Список токенов
        """
        return [token.text for token in tokenize(text)]

    def extract_sentences(self, text: str) -> List[str]:
        """
        Извлекает предложения из текста.

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def count_syllables(self, word: str) -> int:
        """
        Подсчитывает количество слогов в слове.

        Args:
            word: Слово для анализа

        Returns:
            Количество слогов
        """
        vowels: str = "аеёиоуыэюя"
        return sum(1 for char in word.lower() if char in vowels)

    def lemmatize(self, word: str) -> str:
        """
        Лемматизирует слово.

        Args:
            word: Слово для лемматизации

        Returns:
            Лемматизированное слово
        """
        return self.morph.parse(word)[0].normal_form

    def get_pos(self, word: str) -> str:
        """
        Определяет часть речи слова.

        Args:
            word: Слово для анализа

        Returns:
            Часть речи
        """
        p = self.morph.parse(word)[0]
        return p.tag.POS if p.tag.POS is not None else "UNKN"

    def analyze_pos_distribution(self, text: str) -> Dict[str, int]:
        """
        Анализирует распределение частей речи в тексте.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с распределением частей речи
        """
        words = self.tokenize(text)

        pos_counts: Dict[str, int] = {}
        for word in words:
            pos = self.get_pos(word)
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        return pos_counts

    def split_text_into_chunks(
            self, text: str, max_length: int, overlap: int = 50
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
        tokens = [token.text for token in tokenize(text)]
        if len(tokens) <= max_length:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            end_idx = min(start_idx + max_length, len(tokens))

            chunk_text = " ".join(tokens[start_idx:end_idx])
            chunks.append(chunk_text)

            start_idx += max_length - overlap

        return chunks
