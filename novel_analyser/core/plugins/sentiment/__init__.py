"""
Модуль для обработки эмоциональной окраски текста.

Содержит реализации обработчиков эмоциональной окраски текста.
"""

from novel_analyser.core.plugins import create_sentiment_processor
from novel_analyser.core.plugins.sentiment.standard_processor import (
    StandardSentimentProcessor,
)

__all__ = ["StandardSentimentProcessor", "create_sentiment_processor"]
