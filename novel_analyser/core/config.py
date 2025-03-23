"""
Модуль конфигурации для библиотеки анализа текста.
"""

import os
import pathlib
from typing import Dict, Any, Optional, List, Union

import torch
import yaml
from pydantic import BaseModel, Field, model_validator


class OutputConfig(BaseModel):
    """Конфигурация вывода результатов."""

    output_dir: str = Field(
        default="analysis", description="Директория для вывода результатов"
    )
    metrics_file: str = Field(
        default="metrics.txt", description="Имя файла метрик"
    )


class ModelConfig(BaseModel):
    """Конфигурация моделей и устройств."""

    use_gpu: bool = Field(
        default=True, description="Использовать GPU, если доступен"
    )
    device_id: int = Field(default=0, description="ID устройства GPU")
    sentiment_model: str = Field(
        default="cointegrated/rubert-tiny-sentiment-balanced",
        description="Модель для анализа настроений",
    )
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Модель для эмбеддингов",
    )

    @property
    def device(self) -> str:
        """Определяет устройство для вычислений."""
        return "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"


class AnalyseConfig(BaseModel):
    """Общие параметры анализа."""

    max_clusters: int = Field(
        default=20, description="Максимальное количество кластеров"
    )
    min_topics: int = Field(
        default=2, description="Минимальное количество тем"
    )
    max_topics: int = Field(
        default=20, description="Максимальное количество тем"
    )
    reading_speed_wpm: float = Field(
        default=150.0, description="Скорость чтения (слов в минуту)"
    )


class CharacterConfig(BaseModel):
    """Конфигурация анализа персонажей."""

    predefined_names: List[str] = Field(
        default_factory=list, description="Предопределенные имена персонажей"
    )


class SentimentAnalyzeConfig(BaseModel):
    """
    Класс конфигурации для анализа эмоциональной окраски текста.

    Хранит настройки для определения эмоциональной окраски и обработки текста.
    """

    positive_threshold: float = Field(
        default=0.05,
        description="Порог для определения положительного настроения",
    )
    negative_threshold: float = Field(
        default=-0.05,
        description="Порог для определения отрицательного настроения",
    )
    sentiment_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    sentiment_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )


class TopicConfig(BaseModel):
    """Конфигурация тематического моделирования."""

    top_words_count: int = Field(
        default=7, description="Количество топ-слов для каждой темы"
    )
    num_topics_defaults: int = Field(
        default=5, description="Количество тем по умолчанию"
    )
    max_df_vectorizer: float = Field(
        default=1.0,
        description="Максимальная частота документов для векторизатора",
    )
    topic_percentile_lower: int = Field(
        default=10, description="Нижний перцентиль для фильтрации выбросов"
    )
    topic_percentile_upper: int = Field(
        default=90, description="Верхний перцентиль для фильтрации выбросов"
    )


class ClusteringConfig(BaseModel):
    """Конфигурация кластеризации."""

    inertia_threshold: float = Field(
        default=0.1,
        description="Порог инерции для определения числа кластеров",
    )
    pca_components: int = Field(
        default=2, description="Количество компонентов для PCA"
    )
    tsne_components: int = Field(
        default=2, description="Количество компонентов для t-SNE"
    )
    cluster_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    cluster_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )
    perplexity: int = Field(default=30, description="Перплексия для t-SNE")


class RepetitionConfig(BaseModel):
    """Конфигурация анализа повторяемости."""

    high_repetition_threshold: float = Field(
        default=0.7, description="Порог высокой повторяемости"
    )
    medium_repetition_threshold: float = Field(
        default=0.5, description="Порог средней повторяемости"
    )
    top_words_limit: int = Field(default=30, description="Предел для топ-слов")
    top_chart_words: int = Field(
        default=10, description="Количество слов для диаграммы"
    )
    repetition_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    repetition_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )


class NarrativeConfig(BaseModel):
    """Конфигурация анализа нарратива."""

    block_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    block_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )
    sentence_min_length: int = Field(
        default=3, description="Минимальная длина предложения"
    )
    dynamic_rhythm_threshold: int = Field(
        default=5, description="Порог для динамичного ритма"
    )
    moderate_rhythm_threshold: int = Field(
        default=3, description="Порог для умеренного ритма"
    )


class StatisticsConfig(BaseModel):
    """Конфигурация статистического анализа."""

    iqr_factor: float = Field(
        default=0.4, description="Коэффициент межквартильного размаха"
    )
    significance_threshold: float = Field(
        default=0.05, description="Порог значимости"
    )
    filter_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    filter_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )
    cosine_similarity_threshold: float = Field(
        default=0.7, description="Порог косинусной близости"
    )
    quartile_first: int = Field(default=25, description="Первый квартиль")
    quartile_third: int = Field(default=75, description="Третий квартиль")
    significant_value_percentile: int = Field(
        default=40, description="Перцентиль для значимых значений"
    )
    outlier_iqr_multiplier: float = Field(
        default=1.5, description="Множитель IQR для выбросов"
    )
    sturges_offset: int = Field(
        default=1, description="Смещение для правила Стёрджесса"
    )
    friedman_diaconis_factor: int = Field(
        default=2, description="Фактор для правила Фридмана-Диаконса"
    )
    histogram_bins_min: int = Field(
        default=5, description="Минимальное количество бинов"
    )
    histogram_bins_max: int = Field(
        default=50, description="Максимальное количество бинов"
    )


class ParserArgs(BaseModel):
    """Аргументы для парсера текста."""

    character_name_uppercase: bool = Field(
        default=True,
        description="Считать строку в верхнем регистре именем персонажа",
    )
    ignore_html_comments: bool = Field(
        default=True, description="Игнорировать HTML-комментарии"
    )
    exclude_frequent_words_count: int = Field(
        default=30,
        description="Количество наиболее частых слов для исключения",
    )


class ParserConfig(BaseModel):
    """Конфигурация парсера текста."""

    module_path: Optional[str] = Field(
        default=None,
        description="Полный путь к классу парсера в формате 'module.submodule.ParserClass'",
    )
    args: ParserArgs = Field(
        default_factory=ParserArgs,
        description="Аргументы для парсера",
    )


class SentimentProcessorArgs(BaseModel):
    """Аргументы для обработчика эмоциональной окраски."""

    positive_threshold: float = Field(
        default=0.05,
        description="Порог для определения положительного настроения",
    )
    negative_threshold: float = Field(
        default=-0.05,
        description="Порог для определения отрицательного настроения",
    )
    split_chunk_max_length: int = Field(
        default=512, description="Максимальная длина фрагмента текста"
    )
    split_chunk_overlap: int = Field(
        default=50, description="Перекрытие между фрагментами текста"
    )
    sentiment_percentile_lower: int = Field(
        default=5, description="Нижний перцентиль для фильтрации выбросов"
    )
    sentiment_percentile_upper: int = Field(
        default=95, description="Верхний перцентиль для фильтрации выбросов"
    )


class SentimentProcessorConfig(BaseModel):
    """Конфигурация обработчика эмоциональной окраски."""

    module_path: Optional[str] = Field(
        default=None,
        description="Полный путь к классу обработчика в формате 'module.submodule.ProcessorClass'",
    )
    args: SentimentProcessorArgs = Field(
        default_factory=SentimentProcessorArgs,
        description="Аргументы для обработчика",
    )


class EmbeddingProcessorArgs(BaseModel):
    """Аргументы для обработчика эмбеддингов."""

    module_path: Optional[str] = Field(
        default=None,
        description="Название модели для эмбеддингов",
    )
    device: Optional[str] = Field(
        default=None,
        description="Устройство для вычислений ('cuda' или 'cpu')",
    )
    show_progress_bar: bool = Field(
        default=True,
        description="Показывать ли прогресс-бар при кодировании",
    )


class EmbeddingProcessorConfig(BaseModel):
    """Конфигурация обработчика эмбеддингов."""

    module_path: Optional[str] = Field(
        default=None,
        description="Полный путь к классу обработчика в формате 'module.submodule.ProcessorClass'",
    )
    args: EmbeddingProcessorArgs = Field(
        default_factory=EmbeddingProcessorArgs,
        description="Аргументы для обработчика",
    )


class AnalyserConfig(BaseModel):
    """
    Полная конфигурация анализатора текста.
    """

    output: OutputConfig = Field(default_factory=OutputConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    analyse: AnalyseConfig = Field(default_factory=AnalyseConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    character: CharacterConfig = Field(default_factory=CharacterConfig)
    sentiment: SentimentProcessorConfig = Field(
        default_factory=SentimentProcessorConfig
    )
    embedding: EmbeddingProcessorConfig = Field(
        default_factory=EmbeddingProcessorConfig
    )
    sentiment_analyze: SentimentAnalyzeConfig = Field(
        default_factory=SentimentAnalyzeConfig
    )
    topic: TopicConfig = Field(default_factory=TopicConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    repetition: RepetitionConfig = Field(default_factory=RepetitionConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)

    @model_validator(mode="after")
    def create_output_dir(self) -> "AnalyserConfig":
        """Создает директорию для вывода, если она не существует."""
        if not os.path.exists(self.output.output_dir):
            os.makedirs(self.output.output_dir)
        return self

    @classmethod
    def from_yaml_file(
        cls, file_path: Union[str, pathlib.Path]
    ) -> "AnalyserConfig":
        """
        Создает экземпляр класса на основе YAML-файла.

        Args:
            file_path: Путь к YAML-файлу с конфигурацией

        Returns:
            AnalyserConfig: Конфигурация на основе файла
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.model_validate(config_dict or {})

    def as_dict(self) -> Dict[str, Any]:
        """
        Преобразует конфигурацию в словарь.

        Returns:
            Dict[str, Any]: Словарь с конфигурацией
        """
        return self.model_dump()


# Определяем пути к конфигурационным файлам
PACKAGE_DIR = pathlib.Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "configs" / "default_config.yaml"

# Загружаем конфигурацию по умолчанию, если файл существует
if DEFAULT_CONFIG_PATH.exists():
    try:
        default_config = AnalyserConfig.from_yaml_file(DEFAULT_CONFIG_PATH)
    except Exception as e:
        print(f"Ошибка загрузки конфигурации по умолчанию: {e}")
        default_config = AnalyserConfig()
else:
    default_config = AnalyserConfig()


def get_config() -> AnalyserConfig:
    """
    Возвращает текущую глобальную конфигурацию.

    Returns:
        AnalyserConfig: Текущая конфигурация
    """
    return default_config


def configure(new_config: Optional[Dict[str, Any]] = None) -> AnalyserConfig:
    """
    Настраивает глобальную конфигурацию библиотеки.

    Args:
        new_config: Словарь с параметрами конфигурации

    Returns:
        AnalyserConfig: Обновленная конфигурация
    """
    global default_config

    if new_config:
        # Создаем новую конфигурацию, используя обновленные значения
        current_config = default_config.model_dump()

        # Обновляем конфигурацию по секциям, если они указаны
        for section, values in new_config.items():
            if isinstance(values, dict) and section in current_config:
                # Обновляем секцию в существующей конфигурации
                current_config[section].update(values)
            else:
                # Переопределяем целую секцию или добавляем новую
                current_config[section] = values

        # Создаем новую конфигурацию
        default_config = AnalyserConfig.model_validate(current_config)

    return default_config


def load_config_from_file(
    file_path: Union[str, pathlib.Path],
) -> AnalyserConfig:
    """
    Загружает конфигурацию из YAML-файла и устанавливает ее как глобальную.

    Args:
        file_path: Путь к файлу конфигурации

    Returns:
        AnalyserConfig: Загруженная конфигурация
    """
    global default_config
    default_config = AnalyserConfig.from_yaml_file(file_path)
    return default_config
