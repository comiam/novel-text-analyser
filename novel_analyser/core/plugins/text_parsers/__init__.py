"""
Реализации парсеров текста.

Данный модуль содержит различные реализации парсеров текста,
которые можно использовать в библиотеке.
"""

from novel_analyser.core.plugins import create_parser
from novel_analyser.core.plugins.text_parsers.standard_parser import StandardParser

__all__ = ["StandardParser", "create_parser"]
