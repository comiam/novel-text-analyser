"""
Novel Analyser - библиотека для анализа русскоязычных текстов визуальных новелл.
"""

import pathlib

from novel_analyser.analysers.analyser import RootAnalyser
from novel_analyser.core.config import (
    get_config,
    configure,
    AnalyserConfig,
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
    "RootAnalyser",
    "get_config",
    "configure",
    "AnalyserConfig",
]
