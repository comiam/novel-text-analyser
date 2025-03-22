"""
Интерфейсы моделей данных Pydantic.

Модуль содержит базовые классы и протоколы для моделей данных,
используемых в библиотеке.
"""

from typing import Any, Dict, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class PydanticModelProtocol(Protocol):
    """Протокол для моделей Pydantic."""

    def model_dump(self) -> Dict[str, Any]:
        """
        Преобразует модель в словарь.

        Returns:
            Словарь с данными модели
        """
        ...

    def model_dump_json(self, **kwargs) -> str:
        """
        Преобразует модель в JSON-строку.

        Args:
            **kwargs: Аргументы для json.dumps

        Returns:
            JSON-строка с данными модели
        """
        ...

    @classmethod
    def model_validate(cls, obj: Any) -> Any:
        """
        Валидирует объект и создает модель.

        Args:
            obj: Объект для валидации

        Returns:
            Валидированная модель
        """
        ...


T = TypeVar("T", bound=BaseModel)


class ModelFactory:
    """
    Фабрика для создания моделей данных.

    Предоставляет методы для создания и валидации моделей на основе
    входных данных разных типов.
    """

    @staticmethod
    def create_model(model_class: type[T], data: Dict[str, Any] = None) -> T:
        """
        Создает и валидирует экземпляр модели.

        Args:
            model_class: Класс модели для создания
            data: Данные для инициализации модели

        Returns:
            Экземпляр модели
        """
        return model_class.model_validate(data or {})

    @staticmethod
    def validate_model(model: T) -> bool:
        """
        Проверяет, соответствует ли модель своему классу.

        Args:
            model: Модель для проверки

        Returns:
            True, если модель валидна, иначе False
        """
        try:
            model.__class__.model_validate(model.model_dump())
            return True
        except Exception:
            return False

    @staticmethod
    def is_pydantic_model(obj: Any) -> bool:
        """
        Проверяет, является ли объект моделью Pydantic.

        Args:
            obj: Объект для проверки

        Returns:
            True, если объект - модель Pydantic, иначе False
        """
        return isinstance(obj, PydanticModelProtocol)
