"""
Основной модуль для использования из командной строки.
"""

import argparse
import os
import pathlib
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from novel_analyser.api.analyser import TextAnalyser
from novel_analyser.core.config import configure, load_config_from_file
from novel_analyser.utils.logging import configure_logging, get_logger


class CliArguments(BaseModel):
    """Модель для аргументов командной строки."""

    input_file: str = Field(
        ..., description="Путь к файлу с текстом для анализа"
    )
    output: str = Field(
        default="analysis", description="Директория для сохранения результатов"
    )
    analyses: List[str] = Field(
        default=["all"], description="Список анализов для выполнения"
    )
    config: Optional[str] = Field(
        default=None, description="Путь к файлу конфигурации"
    )
    log_config: Optional[str] = Field(
        default=None, description="Путь к файлу конфигурации логирования"
    )
    cpu: bool = Field(default=False, description="Использовать только CPU")


def main():
    """
    Основная функция для запуска анализа из командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Анализ текста визуальных новелл",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "input_file", help="Путь к файлу с текстом для анализа"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Директория для сохранения результатов",
        default="analysis",
    )

    parser.add_argument(
        "--analyses",
        "-a",
        nargs="+",
        choices=[
            "basic",
            "readability",
            "narrative",
            "sentiment",
            "topic",
            "character",
            "repetition",
            "clustering",
            "semantic",
            "all",
        ],
        default=["all"],
        help="""Список анализов для выполнения:
  basic - базовый анализ текста
  readability - анализ читаемости
  narrative - анализ нарративной структуры
  sentiment - анализ эмоциональной окраски
  topic - тематическое моделирование
  character - анализ персонажей
  repetition - анализ повторяемости
  clustering - кластеризация текста
  semantic - семантический анализ
  all - все анализы (по умолчанию)""",
    )

    parser.add_argument(
        "--config",
        "-c",
        help="Путь к файлу конфигурации в формате YAML",
        default=None,
    )

    parser.add_argument(
        "--log-config",
        "-l",
        help="Путь к файлу конфигурации логирования в формате YAML",
        default=None,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Использовать только CPU, даже если доступен GPU",
    )

    args = parser.parse_args()

    # Настраиваем логирование
    # Получаем путь к конфигу логирования
    package_dir = pathlib.Path(__file__).parent.parent
    default_log_config_path = package_dir / "configs" / "logging_config.yaml"

    log_config_path = (
        args.log_config
        if args.log_config
        else (
            default_log_config_path
            if default_log_config_path.exists()
            else None
        )
    )

    configure_logging(log_config_path)

    # Получаем логгер для текущего модуля
    logger = get_logger(__name__)

    logger.info("Запуск анализа текста")

    # Преобразуем аргументы в модель Pydantic
    cli_args = CliArguments(
        input_file=args.input_file,
        output=args.output,
        analyses=args.analyses,
        config=args.config,
        log_config=args.log_config,
        cpu=args.cpu,
    )

    logger.debug(f"Параметры запуска: {cli_args.model_dump()}")

    # Загружаем конфигурацию
    config: Optional[Dict[str, Any]] = None

    # Если указан путь к файлу конфигурации, загружаем его
    if cli_args.config and os.path.exists(cli_args.config):
        try:
            # Определяем формат файла по расширению
            ext = os.path.splitext(cli_args.config)[1].lower()

            if ext in [".yaml", ".yml"]:
                # Загружаем YAML конфигурацию
                load_config_from_file(cli_args.config)
                logger.info(
                    f"Загружена YAML конфигурация из файла: {cli_args.config}"
                )
            else:
                logger.warning(
                    f"Неизвестный формат файла конфигурации: {ext}. Используется конфигурация по умолчанию."
                )
                config = {}
        except Exception as e:
            logger.error(f"Ошибка при загрузке конфигурации из файла: {e}")
            logger.info("Используется конфигурация по умолчанию")
            config = {}
    else:
        config = {}

    # Если указан флаг --cpu, обновляем конфигурацию
    if cli_args.cpu:
        logger.info("Принудительное использование CPU")
        config = config or {}
        config["model"] = config.get("model", {})
        config["model"]["use_gpu"] = False

    # Создаем директорию для результатов, если она не существует
    if not os.path.exists(cli_args.output):
        logger.info(f"Создание директории для результатов: {cli_args.output}")
        os.makedirs(cli_args.output)

    # Устанавливаем директорию для вывода
    config = config or {}
    config["output"] = config.get("output", {})
    config["output"]["output_dir"] = cli_args.output

    # Применяем конфигурацию
    configure(config)

    # Инициализируем анализатор
    logger.info("Инициализация анализатора текста")
    analyser = TextAnalyser()

    # Запускаем анализ
    logger.info(f"Анализ файла: {cli_args.input_file}")
    result = analyser.analyse_file(
        cli_args.input_file, cli_args.output, cli_args.analyses
    )

    # Проверяем, является ли путь абсолютным, если нет - преобразуем в абсолютный
    output_path = (
        cli_args.output
        if os.path.isabs(cli_args.output)
        else os.path.abspath(cli_args.output)
    )

    logger.info(
        f"Анализ успешно выполнен. Результаты сохранены в директории: {output_path}"
    )

    # Выводим краткую сводку
    logger.info("Краткая сводка результатов анализа:")
    summary_lines = result.summary.split("\n")
    # Показываем только первые 20 строк для краткости
    for line in summary_lines[:20]:
        logger.info(line)
    if len(summary_lines) > 20:
        logger.info(f"... и еще {len(summary_lines) - 20} строк")


if __name__ == "__main__":
    main()
