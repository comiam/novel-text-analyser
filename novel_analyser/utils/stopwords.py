"""
Модуль для работы со стоп-словами.
"""

from typing import List

from nltk.corpus import stopwords


def get_stop_words() -> List[str]:
    """
    Возвращает множество стоп-слов для русского языка.

    Returns:
        Множество стоп-слов
    """
    return stopwords.words("russian")
