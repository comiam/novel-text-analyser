"""
Модуль для настройки и управления логированием в приложении.
"""

import logging
import logging.config
import os
from functools import partialmethod
from typing import Dict, Optional, Any

import tqdm
import yaml

# Игнорируем логи от pymorphy3.opencorpora_dict.wrapper, установив уровень WARNING
logging.getLogger("pymorphy3.opencorpora_dict.wrapper").setLevel(
    logging.WARNING
)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
    logging.WARNING
)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
os.environ["TQDM_DISABLE"] = "True"


class AbbreviateNameFilter(logging.Filter):
    """
    Фильтр логирования, сокращающий имя логгера.

    Примеры:
    - 'com.example.myapp.service' -> 'c.e.m.service'
    - 'a.very.long.package.name.MyClass' -> 'a.v.l.p.n.MyClass'
    """

    def filter(self, record):
        name_parts = record.name.split(".")
        if len(name_parts) > 1:
            # Keep the last part unchanged
            abbreviated = [p[0] for p in name_parts[:-1]]
            abbreviated.append(name_parts[-1])
            record.name = ".".join(abbreviated)
        return True


def get_logger(name: str) -> logging.Logger:
    """
    Получает настроенный логгер с указанным именем.

    Args:
        name: Имя логгера, обычно __name__ модуля

    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


def configure_logging(config_path: Optional[str] = None) -> None:
    """
    Настраивает систему логирования из YAML-файла.

    Args:
        config_path: Путь к файлу конфигурации логирования.
                     Если None, используется конфигурация по умолчанию.
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                log_config = yaml.safe_load(f)
                logging.config.dictConfig(log_config)
                logger = get_logger(__name__)
                logger.debug(
                    f"Загружена конфигурация логирования из файла: {config_path}"
                )
        except Exception as e:
            # Fallback to default configuration
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger = get_logger(__name__)
            logger.error(f"Ошибка при загрузке конфигурации логирования: {e}")
    else:
        # Setup default logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = get_logger(__name__)
        logger.debug("Используется конфигурация логирования по умолчанию")


def load_logging_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Загружает конфигурацию логирования из YAML файла.

    Args:
        file_path: Путь к файлу конфигурации

    Returns:
        Dict[str, Any]: Конфигурация логирования

    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: При ошибке парсинга YAML
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
