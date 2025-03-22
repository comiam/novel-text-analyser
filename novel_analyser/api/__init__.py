"""
API-модули для работы с библиотекой анализа текста.
"""

from novel_analyser.api.analyser import TextAnalyser
from novel_analyser.api.character import CharacterAnalyser
from novel_analyser.api.sentiment import SentimentAnalyser
from novel_analyser.api.topic import TopicModeler

__all__ = [
    "TextAnalyser",
    "SentimentAnalyser",
    "TopicModeler",
    "CharacterAnalyser",
]
