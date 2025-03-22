"""
Novel Analyser - библиотека для анализа русскоязычных текстов визуальных новелл.
"""

import pathlib

from novel_analyser.api.analyser import TextAnalyser
from novel_analyser.api.character import CharacterAnalyser
from novel_analyser.api.sentiment import SentimentAnalyser
from novel_analyser.api.topic import TopicModeler
from novel_analyser.core.config import (
    get_config,
    configure,
    AnalyserConfig,
    SentimentAnalyzeConfig,
)
from novel_analyser.utils.logging import configure_logging

# Настраиваем логирование при импорте модуля
package_dir = pathlib.Path(__file__).parent.parent
default_log_config_path = package_dir / "configs" / "logging_config.yaml"

if default_log_config_path.exists():
    configure_logging(str(default_log_config_path))
else:
    configure_logging()

__version__ = "0.1.0"
__all__ = [
    "TextAnalyser",
    "SentimentAnalyser",
    "TopicModeler",
    "CharacterAnalyser",
    "get_config",
    "configure",
    "AnalyserConfig",
    "SentimentAnalyzeConfig",
]
