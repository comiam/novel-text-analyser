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
    create_parser,
    create_sentiment_processor,
    get_parser_registry,
    get_sentiment_processor_registry,
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
    "create_parser",
    "create_sentiment_processor",
    "get_parser_registry",
    "get_sentiment_processor_registry",
]
