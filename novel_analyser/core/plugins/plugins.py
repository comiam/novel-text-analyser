"""
Система управления плагинами для библиотеки анализа текста.

Этот модуль предоставляет функциональность для динамической загрузки
и управления плагинами (парсерами текста, обработчиками эмоциональной
окраски и другими расширяемыми компонентами).
"""

import importlib
from typing import Dict, Optional, Type, TypeVar, Generic, List

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """
    Реестр плагинов определенного типа.

    Обеспечивает загрузку плагинов по имени класса и обработку их
    конфигурационных параметров.
    """

    def __init__(self, base_class: Type[T], default_class: Type[T]):
        """
        Инициализирует реестр плагинов.

        Args:
            base_class: Базовый класс или интерфейс для всех плагинов данного типа
            default_class: Класс плагина, используемый по умолчанию
        """
        self.base_class = base_class
        self.default_class = default_class
        self._registry: Dict[str, Type[T]] = {}

        # Регистрируем класс по умолчанию
        self.register(default_class.__name__, default_class)

    def register(self, name: str, plugin_class: Type[T]) -> None:
        """
        Регистрирует класс плагина в реестре.

        Args:
            name: Имя для регистрации плагина
            plugin_class: Класс плагина

        Raises:
            TypeError: Если плагин не является наследником базового класса
        """
        if not issubclass(plugin_class, self.base_class):
            raise TypeError(
                f"Класс {plugin_class.__name__} не является наследником {self.base_class.__name__}"
            )

        self._registry[name] = plugin_class

    def get_class(self, class_path: Optional[str] = None) -> Type[T]:
        """
        Получает класс плагина из реестра или импортирует его по пути.

        Args:
            class_path: Путь к модулю и классу плагина в формате "module.submodule.ClassName".
                       Если None, то возвращается класс по умолчанию.

        Returns:
            Класс плагина

        Raises:
            ImportError: Если модуль не удалось импортировать
            AttributeError: Если класс не найден в модуле
            TypeError: Если класс не наследует базовый класс
        """
        if not class_path:
            return self.default_class

        # Проверяем, зарегистрирован ли класс
        if class_path in self._registry:
            return self._registry[class_path]

        try:
            # Разбиваем путь на модуль и класс
            module_path, class_name = class_path.rsplit(".", 1)

            # Импортируем модуль
            module = importlib.import_module(module_path)

            # Получаем класс из модуля
            plugin_class = getattr(module, class_name)

            # Проверяем, что класс является наследником базового класса
            if not issubclass(plugin_class, self.base_class):
                raise TypeError(
                    f"Класс {class_name} из модуля {module_path} не наследует {self.base_class.__name__}"
                )

            # Регистрируем класс для будущего использования
            self.register(class_path, plugin_class)

            return plugin_class
        except ImportError as e:
            raise ImportError(f"Не удалось импортировать модуль плагина: {e}")
        except AttributeError as e:
            raise AttributeError(f"Класс плагина не найден в модуле: {e}")

    def create_instance(self, class_path: Optional[str] = None, **kwargs) -> T:
        """
        Создает экземпляр плагина.

        Args:
            class_path: Путь к модулю и классу плагина в формате "module.submodule.ClassName".
                       Если None, то используется класс по умолчанию.
            **kwargs: Аргументы для инициализации плагина.

        Returns:
            Экземпляр плагина
        """
        plugin_class = self.get_class(class_path)
        return plugin_class(**kwargs)

    def list_registered(self) -> List[str]:
        """
        Возвращает список зарегистрированных плагинов.

        Returns:
            Список имен зарегистрированных плагинов
        """
        return list(self._registry.keys())


# Глобальные реестры для различных типов плагинов
_parser_registry = None
_sentiment_processor_registry = None
_embedding_encoder_registry = None


def get_parser_registry():
    """
    Получает реестр парсеров текста.

    Returns:
        Реестр парсеров текста
    """
    global _parser_registry

    if _parser_registry is None:
        # Import here to avoid circular imports
        from novel_analyser.core.interfaces.parser import BaseParser
        from novel_analyser.core.plugins.text_parsers.standard_parser import (
            StandardParser,
        )

        _parser_registry = PluginRegistry(BaseParser, StandardParser)

    return _parser_registry


def get_sentiment_processor_registry():
    """
    Получает реестр обработчиков эмоциональной окраски.

    Returns:
        Реестр обработчиков эмоциональной окраски
    """
    global _sentiment_processor_registry

    if _sentiment_processor_registry is None:
        # Import here to avoid circular imports
        from novel_analyser.core.interfaces.sentiment import (
            BaseSentimentProcessor,
        )
        from novel_analyser.core.plugins.sentiment.standard_processor import (
            StandardSentimentProcessor,
        )

        _sentiment_processor_registry = PluginRegistry(
            BaseSentimentProcessor, StandardSentimentProcessor
        )

    return _sentiment_processor_registry


def get_embedding_encoder_registry():
    """
    Получает реестр обработчиков эмбеддингов.

    Returns:
        Реестр обработчиков эмбеддингов
    """
    global _embedding_encoder_registry

    if _embedding_encoder_registry is None:
        # Import here to avoid circular imports
        from novel_analyser.core.interfaces.embedding import (
            BaseEmbeddingEncoder,
        )
        from novel_analyser.core.plugins.embedding.standard_processor import (
            StandardEmbeddingProcessor,
        )

        _embedding_encoder_registry = PluginRegistry(
            BaseEmbeddingEncoder, StandardEmbeddingProcessor
        )

    return _embedding_encoder_registry


def create_text_parser():
    """
    Создает экземпляр парсера текста на основе конфигурации.

    Returns:
        Экземпляр парсера текста
    """
    # Import here to avoid circular imports
    from novel_analyser.core.config import get_config

    config = get_config()
    module_path_path = (
        config.parser.module_path
        if hasattr(config.parser, "module_path")
        else None
    )

    parser_args = (
        config.parser.args.model_dump()
        if hasattr(config.parser, "args")
        else {}
    )

    return get_parser_registry().create_instance(
        module_path_path, **parser_args
    )


def create_sentiment_processor():
    """
    Создает экземпляр обработчика эмоциональной окраски на основе конфигурации.

    Returns:
        Экземпляр обработчика эмоциональной окраски
    """
    # Import here to avoid circular imports
    from novel_analyser.core.config import get_config

    config = get_config()
    module_path_path = (
        config.sentiment.module_path
        if hasattr(config.sentiment, "module_path")
        else None
    )

    processor_args = (
        config.sentiment.args.model_dump()
        if hasattr(config.sentiment, "args")
        else {}
    )

    return get_sentiment_processor_registry().create_instance(
        module_path_path, **processor_args
    )


def create_embedding_encoder():
    """
    Создает экземпляр обработчика эмбеддингов на основе конфигурации.

    Returns:
        Экземпляр обработчика эмбеддингов
    """
    # Import here to avoid circular imports
    from novel_analyser.core.config import get_config

    config = get_config()
    module_path_path = (
        config.embedding.module_path
        if hasattr(config.embedding, "module_path")
        else None
    )

    processor_args = (
        config.embedding.args.model_dump()
        if hasattr(config.embedding, "args")
        else {}
    )

    return get_embedding_encoder_registry().create_instance(
        module_path_path, **processor_args
    )
