"""
Модуль для семантического анализа текста.
"""

import os
from typing import List, Optional

import numpy as np

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.interfaces.embedding import BaseEmbeddingEncoder
from novel_analyser.core.interfaces.parser import BaseParser
from novel_analyser.utils.plot import save_histogram
from novel_analyser.utils.stat import compute_average_cosine_similarity


class SemanticAnalyser(BaseAnalyser):
    """
    Класс для анализа семантической связности и когерентности текста.
    """

    def __init__(
        self,
        text_parser: BaseParser,
        embedding_encoder: BaseEmbeddingEncoder,
        config: Optional[AnalyserConfig] = None,
    ):
        """
        Инициализирует анализатор семантики.

        Args:
            text_parser: Парсер для извлечения текстовых блоков
            embedding_encoder: Процессор для получения эмбеддингов
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.text_parser = text_parser
        self.embedding_encoder = embedding_encoder

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет семантический анализ текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Извлекаем предложения из блоков
        sentences = self.text_parser.extract_all_sentences(blocks)

        if not sentences:
            result.summary = "Семантический анализ:\n  Не найдено предложений для анализа.\n"
            return result

        # Получаем эмбеддинги для предложений
        sentence_embeddings = self.embedding_encoder.encode(sentences)

        # Вычисляем когерентность (среднюю косинусную близость между соседними предложениями)
        avg_cosine = compute_average_cosine_similarity(sentence_embeddings)

        # Вычисляем косинусную близость между всеми парами последовательных предложений
        coherence_scores = []

        for i in range(len(sentence_embeddings) - 1):
            v1 = sentence_embeddings[i]
            v2 = sentence_embeddings[i + 1]

            cos_sim = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9
            )
            coherence_scores.append(cos_sim)

        # Сохраняем гистограмму когерентности
        save_histogram(
            coherence_scores,
            "Когерентность между соседними предложениями",
            "Косинусная близость",
            "Количество пар предложений",
            self.save_figure("coherence_histogram.png"),
        )

        # Оцениваем семантическую стабильность текста
        semantic_stability = self.assess_semantic_stability(
            sentence_embeddings
        )

        # Заполняем метрики
        result.metrics.update(
            {
                "avg_coherence": float(avg_cosine),
                "coherence_scores": [
                    float(score) for score in coherence_scores
                ],
                "semantic_stability": float(semantic_stability),
            }
        )

        # Заполняем пути к сохраненным изображениям
        result.figures.update(
            {
                "coherence_histogram": os.path.join(
                    self.config.output.output_dir, "coherence_histogram.png"
                )
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Семантический анализ:\n"
        result.summary += f"  Средняя когерентность между соседними предложениями: {avg_cosine:.3f}\n"
        result.summary += (
            f"  Семантическая стабильность текста: {semantic_stability:.3f}\n"
        )

        # Добавляем интерпретацию
        if avg_cosine > 0.8:
            result.summary += "  Высокая семантическая связность текста. Повествование логично и последовательно.\n"
        elif avg_cosine > 0.6:
            result.summary += "  Средняя семантическая связность текста. В целом повествование последовательно.\n"
        else:
            result.summary += "  Низкая семантическая связность текста. Возможны скачки или непоследовательность в повествовании.\n"

        if semantic_stability > 0.8:
            result.summary += "  Высокая семантическая стабильность. Текст сохраняет основную тему на протяжении всего повествования.\n"
        elif semantic_stability > 0.6:
            result.summary += "  Средняя семантическая стабильность. Текст в целом следует основной теме, с небольшими отступлениями.\n"
        else:
            result.summary += "  Низкая семантическая стабильность. Текст может содержать значительные отступления от основной темы.\n"

        return result

    def assess_semantic_stability(self, embeddings: np.ndarray) -> float:
        """
        Оценивает семантическую стабильность текста, сравнивая первую и вторую половины текста.

        Args:
            embeddings: Массив эмбеддингов предложений

        Returns:
            Мера семантической стабильности (от 0 до 1)
        """
        if len(embeddings) < 4:
            return 1.0  # Недостаточно данных для анализа

        # Разделяем текст на две примерно равные части
        mid_point = len(embeddings) // 2
        first_half = embeddings[:mid_point]
        second_half = embeddings[mid_point:]

        # Вычисляем средние векторы для каждой половины
        first_half_mean = np.mean(first_half, axis=0)
        second_half_mean = np.mean(second_half, axis=0)

        # Нормализуем векторы
        first_half_norm = first_half_mean / (
            np.linalg.norm(first_half_mean) + 1e-9
        )
        second_half_norm = second_half_mean / (
            np.linalg.norm(second_half_mean) + 1e-9
        )

        # Вычисляем косинусную близость между половинами текста
        cos_sim = np.dot(first_half_norm, second_half_norm)

        return float(cos_sim)
