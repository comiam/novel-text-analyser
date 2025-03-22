"""
Система управления плагинами для библиотеки анализа текста.

Этот модуль предоставляет функциональность для динамической загрузки
и управления плагинами (парсерами текста, обработчиками эмоциональной
окраски и другими расширяемыми компонентами).
"""

import importlib
from typing import Dict, Optional, Type, TypeVar, Generic, List, Any

T = TypeVar("T")


class PluginRegistry(Generic[T]):
    """
    Реестр плагинов определенного типа.

    Обеспечивает загрузку плагинов по имени класса и обработку их
    конфигурационных параметров.
    """

    def __init__(self, base_class: Type[T], default_class_path: str):
        """
        Инициализирует реестр плагинов.

        Args:
            base_class: Базовый класс или интерфейс для всех плагинов данного типа
            default_class_path: Путь к классу плагина, используемому по умолчанию
        """
        self.base_class = base_class
        self.default_class_path = default_class_path
        self._registry: Dict[str, Type[T]] = {}
        self._default_class = None

    @property
    def default_class(self) -> Type[T]:
        """Получает класс плагина по умолчанию с ленивой загрузкой."""
        if self._default_class is None:
            self._default_class = self.get_class(self.default_class_path)
        return self._default_class

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


def _lazy_import(import_path: str) -> Any:
    """
    Выполняет ленивый импорт модуля или класса.

    Args:
        import_path: Путь к импортируемому объекту

    Returns:
        Импортированный объект
    """
    module_path, obj_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def get_parser_registry():
    """
    Получает реестр парсеров текста.

    Returns:
        Реестр парсеров текста
    """
    global _parser_registry

    if _parser_registry is None:
        # Lazy import to avoid circular imports
        BaseParser = _lazy_import(
            "novel_analyser.interfaces.parser.BaseParser"
        )
        _parser_registry = PluginRegistry(
            BaseParser, "novel_analyser.parsers.standard_parser.StandardParser"
        )

    return _parser_registry


def get_sentiment_processor_registry():
    """
    Получает реестр обработчиков эмоциональной окраски.

    Returns:
        Реестр обработчиков эмоциональной окраски
    """
    global _sentiment_processor_registry

    if _sentiment_processor_registry is None:
        # Lazy import to avoid circular imports
        BaseSentimentProcessor = _lazy_import(
            "novel_analyser.interfaces.sentiment.BaseSentimentProcessor"
        )
        _sentiment_processor_registry = PluginRegistry(
            BaseSentimentProcessor,
            "novel_analyser.sentiment.standard_processor.StandardSentimentProcessor",
        )

    return _sentiment_processor_registry


def create_parser():
    """
    Создает экземпляр парсера текста на основе конфигурации.

    Returns:
        Экземпляр парсера текста
    """
    # Import here to avoid circular imports
    from novel_analyser.core.config import get_config

    config = get_config()
    parser_class_path = (
        config.parser.parser_class
        if hasattr(config.parser, "parser_class")
        else None
    )

    parser_args = (
        config.parser.args.model_dump()
        if hasattr(config.parser, "args")
        else {}
    )

    return get_parser_registry().create_instance(
        parser_class_path, **parser_args
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
    processor_class_path = (
        config.sentiment.processor_class
        if hasattr(config.sentiment, "processor_class")
        else None
    )

    processor_args = (
        config.sentiment.args.model_dump()
        if hasattr(config.sentiment, "args")
        else {}
    )

    return get_sentiment_processor_registry().create_instance(
        processor_class_path, **processor_args
    )
