"""
Основной API модуль для анализа текста.
"""

import os
from typing import Dict, List, Optional, Any, Literal

from pydantic import BaseModel, Field

from novel_analyser.analysers.basic import BasicAnalyser
from novel_analyser.analysers.character import CharacterAnalyser
from novel_analyser.analysers.clustering import ClusteringAnalyser
from novel_analyser.analysers.narrative import NarrativeAnalyser
from novel_analyser.analysers.readability import ReadabilityAnalyser
from novel_analyser.analysers.repetition import RepetitionAnalyser
from novel_analyser.analysers.semantic import SemanticAnalyser
from novel_analyser.analysers.sentiment import SentimentAnalyser
from novel_analyser.analysers.topic import TopicAnalyser
from novel_analyser.core.base_analyser import AnalysisResult
from novel_analyser.core.config import get_config, configure
from novel_analyser.core.plugins import (
    create_sentiment_processor,
    create_text_parser,
)
from novel_analyser.core.plugins.plugins import create_embedding_encoder
from novel_analyser.core.text_processor import TextProcessor
from novel_analyser.utils.logging import get_logger

logger = get_logger(__name__)


class AnalysisOptions(BaseModel):
    """Модель для опций анализа текста."""

    analyses: List[
        Literal[
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
        ]
    ] = Field(default=["all"], description="Список анализов для выполнения")
    output_dir: Optional[str] = Field(
        default=None, description="Директория для вывода результатов"
    )


class RootAnalyser:
    """
    Главный класс для анализа текста.
    Предоставляет единый интерфейс для всех видов анализа.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализирует анализатор текста.

        Args:
            config: Словарь с параметрами конфигурации.
                   Если None, используется конфигурация по умолчанию.
        """
        if config:
            configure(config)
        self.config = get_config()

        # Создаем парсер для анализа текста
        logger.debug("Инициализация парсера текста")
        self.parser = create_text_parser()
        self.text_processor = TextProcessor()
        self.text_parser = create_text_parser()
        self.sentiment_processor = create_sentiment_processor()
        self.embedding_encoder = create_embedding_encoder()

        # Инициализируем все анализаторы
        logger.debug("Инициализация анализаторов")
        self.basic_analyser = BasicAnalyser(self.config)
        self.readability_analyser = ReadabilityAnalyser(
            self.text_processor, self.config
        )
        self.narrative_analyser = NarrativeAnalyser(
            self.text_processor, self.config
        )
        self.sentiment_analyser = SentimentAnalyser(
            self.sentiment_processor, self.config
        )
        self.topic_analyser = TopicAnalyser(self.config)
        self.character_analyser = CharacterAnalyser(
            self.text_parser,
            self.sentiment_analyser,
            self.basic_analyser,
            self.config,
        )
        self.repetition_analyser = RepetitionAnalyser(
            self.text_processor, self.config
        )
        self.clustering_analyser = ClusteringAnalyser(
            self.text_parser, self.embedding_encoder, self.config
        )
        self.semantic_analyser = SemanticAnalyser(
            self.text_parser, self.embedding_encoder, self.config
        )
        logger.debug("Все анализаторы инициализированы")

    def analyse_file(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        analyses: Optional[List[str]] = None,
    ) -> AnalysisResult:
        """
        Анализирует текст из файла.

        Args:
            file_path: Путь к файлу с текстом
            output_dir: Директория для сохранения результатов.
                       Если None, используется директория из конфигурации.
            analyses: Список анализов для выполнения.
                     Если None, выполняются все анализы.
                     Доступные значения: 'basic', 'readability', 'narrative',
                     'sentiment', 'topic', 'character', 'repetition',
                     'clustering', 'semantic', 'all'

        Returns:
            Результат анализа
        """
        options = AnalysisOptions(
            analyses=analyses or ["all"], output_dir=output_dir
        )

        if options.output_dir:
            self.config.output.output_dir = options.output_dir
            if not os.path.exists(options.output_dir):
                logger.debug(
                    f"Создание директории для результатов: {options.output_dir}"
                )
                os.makedirs(options.output_dir)

        logger.debug(f"Чтение файла: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(
                f"Файл успешно прочитан, размер: {len(content)} символов"
            )
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {str(e)}")
            raise

        return self.analyse_text(content, options.analyses)

    def analyse_text(
        self, text: str, analyses: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Анализирует текст.

        Args:
            text: Текст для анализа
            analyses: Список анализов для выполнения.
                     Если None, выполняются все анализы.
                     Доступные значения: 'basic', 'readability', 'narrative',
                     'sentiment', 'topic', 'character', 'repetition',
                     'clustering', 'semantic', 'all'

        Returns:
            Результат анализа
        """
        options = AnalysisOptions(analyses=analyses or ["all"])
        logger.info(
            f"Начало анализа текста, выбранные анализы: {options.analyses}"
        )

        # Преобразуем список анализов к структуре для обхода
        all_analyses = {
            "basic": self.basic_analyser,
            "readability": self.readability_analyser,
            "narrative": self.narrative_analyser,
            "sentiment": self.sentiment_analyser,
            "topic": self.topic_analyser,
            "character": self.character_analyser,
            "repetition": self.repetition_analyser,
            "clustering": self.clustering_analyser,
            "semantic": self.semantic_analyser,
        }

        if "all" in options.analyses:
            logger.debug("Выбраны все доступные анализы")
            selected_analyses = all_analyses
        else:
            selected_analyses = {
                name: analyser
                for name, analyser in all_analyses.items()
                if name in options.analyses
            }
            logger.debug(
                f"Выбранные анализаторы: {', '.join(selected_analyses.keys())}"
            )

        # Парсим блоки текста
        logger.debug("Парсинг блоков текста")
        blocks = self.parser.parse_blocks(text)
        logger.debug(f"Получено {len(blocks)} блоков текста")

        logger.debug("Парсинг сырых блоков текста")
        raw_blocks = self.parser.parse_blocks(text, raw_style=True)
        logger.debug(f"Получено {len(raw_blocks)} сырых блоков текста")

        # Создаем общий результат
        result = AnalysisResult()

        # Выполняем выбранные анализы
        for name, analyser in selected_analyses.items():
            logger.info(f"Выполнение анализа: {name}")
            try:
                match name:
                    case "basic":
                        # Для базового анализа передаем и очищенные, и сырые блоки
                        analysis_result = analyser.analyse(blocks, raw_blocks)
                    case "character":
                        # Для анализа персонажей передаем только сырые блоки
                        analysis_result = analyser.analyse(raw_blocks)
                    case _:
                        # Для остальных анализов передаем только очищенные блоки
                        analysis_result = analyser.analyse(blocks)

                # Добавляем результаты к общему результату
                result.extend(analysis_result)
                logger.debug(
                    f"Получено {len(analysis_result.metrics)} метрик и {len(analysis_result.figures)} визуализаций"
                )
            except Exception as e:
                logger.error(f"Ошибка при выполнении анализа {name}: {str(e)}")

        # Сохраняем метрики в файл
        metrics_path = os.path.join(
            self.config.output.output_dir, self.config.output.metrics_file
        )
        logger.debug(f"Сохранение метрик в файл: {metrics_path}")
        try:
            result.save_metrics(metrics_path)
            logger.info("Метрики успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка при сохранении метрик: {str(e)}")

        return result

    def get_available_analyses(self) -> List[str]:
        """
        Возвращает список доступных анализов.

        Returns:
            Список названий доступных анализов
        """
        return [
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
        ]
