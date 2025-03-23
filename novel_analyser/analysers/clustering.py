"""
Модуль для кластеризации текста на основе семантических эмбеддингов.
"""

import os
from typing import List, Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.interfaces.embedding import BaseEmbeddingEncoder
from novel_analyser.core.interfaces.parser import BaseParser
from novel_analyser.utils.plot import save_elbow_curve, save_embedding_scatter


class ClusteringAnalyser(BaseAnalyser):
    """
    Класс для кластеризации текста на основе семантических эмбеддингов.
    """

    def __init__(
        self,
        text_parser: BaseParser,
        embedding_encoder: BaseEmbeddingEncoder,
        config: Optional[AnalyserConfig] = None,
    ):
        """
        Инициализирует анализатор кластеризации.

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
        Выполняет кластеризацию текстовых блоков по эмбеддингам.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Получаем эмбеддинги для блоков
        block_embeddings = self.embedding_encoder.encode(blocks)

        # Извлекаем предложения для кластеризации
        sentences = self.text_parser.extract_all_sentences(blocks)

        # Кластеризуем блоки
        (
            optimal_k_blocks,
            block_labels,
            block_emb_pca,
            block_emb_tsne,
            inertias_blocks,
        ) = self.cluster_embeddings(
            block_embeddings,
            max_clusters=self.config.analyse.max_clusters,
            threshold=self.config.clustering.inertia_threshold,
        )

        # Сохраняем графики для блоков
        save_elbow_curve(
            inertias_blocks,
            self.config.analyse.max_clusters,
            self.save_figure("block_elbow_curve.png"),
        )

        save_embedding_scatter(
            block_emb_pca,
            block_labels,
            f"Кластеризация блоков (PCA, {optimal_k_blocks} кластеров)",
            self.save_figure("block_clusters_pca.png"),
        )

        save_embedding_scatter(
            block_emb_tsne,
            block_labels,
            f"Кластеризация блоков (t-SNE, {optimal_k_blocks} кластеров)",
            self.save_figure("block_clusters_tsne.png"),
        )

        # Добавляем информацию о блоках в метрики
        result.metrics.update(
            {
                "optimal_num_clusters_blocks": optimal_k_blocks,
                "block_cluster_labels": (
                    block_labels.tolist()
                    if len(set(block_labels)) > 1
                    else f"все элементы из кластера {block_labels[0]}"
                ),
            }
        )

        # Добавляем пути к сохраненным графикам
        result.figures.update(
            {
                "block_elbow_curve": os.path.join(
                    self.config.output.output_dir, "block_elbow_curve.png"
                ),
                "block_clusters_pca": os.path.join(
                    self.config.output.output_dir, "block_clusters_pca.png"
                ),
                "block_clusters_tsne": os.path.join(
                    self.config.output.output_dir, "block_clusters_tsne.png"
                ),
            }
        )

        # Если есть предложения, кластеризуем их тоже
        if sentences:
            sentence_embeddings = self.embedding_encoder.encode(sentences)

            (
                optimal_k_sent,
                sentence_labels,
                sent_emb_pca,
                sent_emb_tsne,
                inertias_sent,
            ) = self.cluster_embeddings(
                sentence_embeddings,
                max_clusters=self.config.analyse.max_clusters,
                threshold=self.config.clustering.inertia_threshold,
            )

            # Сохраняем графики для предложений
            save_elbow_curve(
                inertias_sent,
                self.config.analyse.max_clusters,
                self.save_figure("sentence_elbow_curve.png"),
            )

            save_embedding_scatter(
                sent_emb_pca,
                sentence_labels,
                f"Кластеризация предложений (PCA, {optimal_k_sent} кластеров)",
                self.save_figure("sentence_clusters_pca.png"),
            )

            save_embedding_scatter(
                sent_emb_tsne,
                sentence_labels,
                f"Кластеризация предложений (t-SNE, {optimal_k_sent} кластеров)",
                self.save_figure("sentence_clusters_tsne.png"),
            )

            # Добавляем информацию о предложениях в метрики
            result.metrics.update(
                {
                    "optimal_num_clusters_sentences": optimal_k_sent,
                    "sentence_cluster_labels": (
                        sentence_labels.tolist()
                        if len(set(sentence_labels)) > 1
                        else f"все элементы из кластера {sentence_labels[0]}"
                    ),
                }
            )

            # Добавляем пути к сохраненным графикам
            result.figures.update(
                {
                    "sentence_elbow_curve": os.path.join(
                        self.config.output.output_dir,
                        "sentence_elbow_curve.png",
                    ),
                    "sentence_clusters_pca": os.path.join(
                        self.config.output.output_dir,
                        "sentence_clusters_pca.png",
                    ),
                    "sentence_clusters_tsne": os.path.join(
                        self.config.output.output_dir,
                        "sentence_clusters_tsne.png",
                    ),
                }
            )

        # Заполняем текстовое резюме
        result.summary = "Кластеризация на основе семантических эмбеддингов:\n"
        result.summary += (
            f"  Оптимальное число кластеров для блоков: {optimal_k_blocks}\n"
        )

        # Добавляем информацию о распределении блоков по кластерам
        cluster_sizes = np.bincount(block_labels)
        for i in range(optimal_k_blocks):
            result.summary += f"  Кластер {i + 1}: {cluster_sizes[i]} блоков ({cluster_sizes[i] / len(blocks) * 100:.1f}%)\n"

        if sentences:
            result.summary += f"\n  Оптимальное число кластеров для предложений: {optimal_k_sent}\n"

            # Добавляем информацию о распределении предложений по кластерам, если они есть
            sent_cluster_sizes = np.bincount(sentence_labels)
            result.summary += "  Распределение предложений по кластерам:\n"
            for i in range(optimal_k_sent):
                result.summary += f"    Кластер {i + 1}: {sent_cluster_sizes[i]} предложений ({sent_cluster_sizes[i] / len(sentences) * 100:.1f}%)\n"

        return result

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """
        Находит оптимальное количество кластеров для KMeans кластеризации на основе инерции.

        Args:
            embeddings: Массив эмбеддингов для кластеризации
            max_clusters: Максимальное количество кластеров для проверки
            threshold: Пороговое значение для определения оптимального количества кластеров

        Returns:
            Оптимальное количество кластеров
        """
        if max_clusters is None:
            max_clusters = self.config.analyse.max_clusters

        if threshold is None:
            threshold = self.config.clustering.inertia_threshold

        inertias: List[float] = []

        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        optimal_k: int = max_clusters

        for i in range(1, len(inertias)):
            drop: float = (inertias[i - 1] - inertias[i]) / inertias[i - 1]

            if drop < threshold:
                optimal_k = i
                break

        return optimal_k

    def reduce_dimensions_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Уменьшает размерность векторов с помощью метода главных компонент (PCA).

        Args:
            embeddings: Входной массив векторов

        Returns:
            Массив векторов с уменьшенной до 2 размерностью
        """
        pca = PCA(
            n_components=self.config.clustering.pca_components, random_state=42
        )
        return pca.fit_transform(embeddings)

    def reduce_dimensions_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Уменьшает размерность векторов с помощью t-SNE.

        Args:
            embeddings: Входные векторы для уменьшения размерности

        Returns:
            Векторы, уменьшенные до 2-х измерений
        """
        tsne = TSNE(
            n_components=self.config.clustering.tsne_components,
            perplexity=self.config.clustering.perplexity,
            random_state=42,
        )
        return tsne.fit_transform(embeddings)

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 10,
        threshold: float = 0.1,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, List[float]]:
        """
        Кластеризация эмбеддингов с использованием KMeans и определение оптимального количества кластеров.

        Args:
            embeddings: Массив эмбеддингов для кластеризации
            max_clusters: Максимальное количество кластеров для рассмотрения
            threshold: Порог для определения оптимального количества кластеров

        Returns:
            Кортеж из:
            - Оптимальное количество кластеров
            - Метки кластеров для каждого эмбеддинга
            - Эмбеддинги, уменьшенные до 2D с помощью PCA
            - Эмбеддинги, уменьшенные до 2D с помощью t-SNE
            - Список значений инерции для каждого количества кластеров
        """
        inertias: List[float] = []

        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        optimal_k: int = self.find_optimal_clusters(
            embeddings, max_clusters, threshold
        )

        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
        labels: np.ndarray = kmeans_final.fit_predict(embeddings)

        pca_reduced: np.ndarray = self.reduce_dimensions_pca(embeddings)
        tsne_reduced: np.ndarray = self.reduce_dimensions_tsne(embeddings)

        return optimal_k, labels, pca_reduced, tsne_reduced, inertias
