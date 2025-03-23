"""
Ядро библиотеки для анализа текста.
"""

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import (
    AnalyserConfig,
    get_config,
    configure,
    SentimentAnalyzeConfig,
)
from novel_analyser.core.plugins import (
    create_text_parser,
    create_sentiment_processor,
    create_embedding_encoder,
    get_parser_registry,
    get_sentiment_processor_registry,
    get_embedding_encoder_registry,
)

__all__ = [
    # Базовые классы и конфигурация
    "AnalyserConfig",
    "get_config",
    "configure",
    "BaseAnalyser",
    "AnalysisResult",
    # Конфигурация анализа настроений
    "SentimentAnalyzeConfig",
    # Управление плагинами
    "create_text_parser",
    "create_sentiment_processor",
    "create_embedding_encoder",
    "get_parser_registry",
    "get_sentiment_processor_registry",
    "get_embedding_encoder_registry",
]
