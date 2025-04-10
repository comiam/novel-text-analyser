"""
Модуль с интерфейсами для подключаемых компонентов библиотеки.

Содержит базовые абстрактные классы и протоколы, которые должны реализовывать
все подключаемые модули.
"""

from novel_analyser.core.interfaces.embedding import (
    BaseEmbeddingEncoder,
    EmbeddingProcessorProtocol,
)
from novel_analyser.core.interfaces.parser import BaseParser, ParserProtocol
from novel_analyser.core.interfaces.sentiment import (
    BaseSentimentProcessor,
    SentimentProcessorProtocol,
)

__all__ = [
    "BaseParser",
    "ParserProtocol",
    "BaseSentimentProcessor",
    "SentimentProcessorProtocol",
    "BaseEmbeddingEncoder",
    "EmbeddingProcessorProtocol",
]
