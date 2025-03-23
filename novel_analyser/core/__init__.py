"""
Ядро библиотеки для анализа текста.
"""

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import (
    AnalyserConfig,
    get_config,
    configure,
)

__all__ = [
    # Базовые классы и конфигурация
    "AnalyserConfig",
    "get_config",
    "configure",
    "BaseAnalyser",
    "AnalysisResult",
]
