"""
Модуль для обработки эмбеддингов текста.

Содержит реализации обработчиков эмбеддингов текста.
"""

from novel_analyser.core.plugins import create_embedding_encoder
from novel_analyser.core.plugins.embedding.standard_processor import (
    StandardEmbeddingProcessor,
)

__all__ = ["StandardEmbeddingProcessor", "create_embedding_encoder"]
