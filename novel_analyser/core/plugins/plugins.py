"""
Система управления плагинами для библиотеки анализа текста.

Этот модуль предоставляет функциональность для динамической загрузки
и управления плагинами (парсерами текста, обработчиками эмоциональной
окраски и другими расширяемыми компонентами).
"""

import importlib
from typing import Dict, Optional, Type, TypeVar, Generic, List
from novel_analyser.core.config import get_config

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
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            # Проверяем, что класс является наследником базового класса
            if not issubclass(plugin_class, self.base_class):
                raise TypeError(
                    f"Класс {class_name} из модуля {module_path} не наследует {self.base_class.__name__}"
                )

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


# Кеш для реестров плагинов
_plugin_registries = {}


def _get_registry(
    base_class_path: str, default_class_path: str
) -> PluginRegistry:
    """
    Получает или создает реестр плагинов для указанных базового и дефолтного классов.

    Args:
        base_class_path: Путь к базовому классу
        default_class_path: Путь к классу по умолчанию

    Returns:
        Реестр плагинов
    """
    registry_key = f"{base_class_path}:{default_class_path}"

    if registry_key not in _plugin_registries:
        # Импортируем классы для реестра
        module_path, class_name = base_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        base_class = getattr(module, class_name)

        module_path, class_name = default_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        default_class = getattr(module, class_name)

        # Создаем реестр и сохраняем его в кеше
        _plugin_registries[registry_key] = PluginRegistry(
            base_class, default_class
        )

    return _plugin_registries[registry_key]


def get_parser_registry():
    """
    Получает реестр парсеров текста.

    Returns:
        Реестр парсеров текста
    """
    return _get_registry(
        "novel_analyser.core.interfaces.parser.BaseParser",
        "novel_analyser.core.plugins.text_parsers.standard_parser.StandardParser",
    )


def get_sentiment_processor_registry():
    """
    Получает реестр обработчиков эмоциональной окраски.

    Returns:
        Реестр обработчиков эмоциональной окраски
    """
    return _get_registry(
        "novel_analyser.core.interfaces.sentiment.BaseSentimentProcessor",
        "novel_analyser.core.plugins.sentiment.standard_processor.StandardSentimentProcessor",
    )


def get_embedding_encoder_registry():
    """
    Получает реестр обработчиков эмбеддингов.

    Returns:
        Реестр обработчиков эмбеддингов
    """
    return _get_registry(
        "novel_analyser.core.interfaces.embedding.BaseEmbeddingEncoder",
        "novel_analyser.core.plugins.embedding.standard_processor.StandardEmbeddingProcessor",
    )


def _create_instance_by_registry(
    registry: PluginRegistry, config_name: str
) -> T:
    """
    Создает экземпляр модуля из реестра плагинов на основе конфигурации.
    Функция получает конфигурацию по указанному имени, извлекает путь к модулю
    и аргументы, необходимые для его создания, затем создает экземпляр через реестр.

    Args:
        registry (PluginRegistry): Реестр плагинов, используемый для создания экземпляра.
        config_name (str): Имя конфигурации в общих настройках.

    Returns:
        T: Созданный экземпляр модуля указанного типа.

    Note:
        Функция ожидает, что в конфигурации будут поля 'module_path' и опционально 'args'.
        Если 'args' имеет метод model_dump(), он будет вызван для получения аргументов в виде словаря.
    """

    config = get_config()
    module_config = getattr(config, config_name, None)
    module_path = getattr(module_config, "module_path", None)
    parser_args = getattr(module_config, "args", {})

    if hasattr(parser_args, "model_dump"):
        parser_args = parser_args.model_dump()

    return registry.create_instance(module_path, **parser_args)


def create_text_parser():
    """
    Создает экземпляр парсера текста на основе конфигурации.

    Returns:
        Экземпляр парсера текста
    """
    return _create_instance_by_registry(get_parser_registry(), "parser")


def create_sentiment_processor():
    """
    Создает экземпляр обработчика эмоциональной окраски на основе конфигурации.

    Returns:
        Экземпляр обработчика эмоциональной окраски
    """
    return _create_instance_by_registry(
        get_sentiment_processor_registry(), "sentiment"
    )


def create_embedding_encoder():
    """
    Создает экземпляр обработчика эмбеддингов на основе конфигурации.

    Returns:
        Экземпляр обработчика эмбеддингов
    """
    return _create_instance_by_registry(
        get_embedding_encoder_registry(), "embedding"
    )
