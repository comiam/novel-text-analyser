"""
Стандартная реализация обработчика эмбеддингов текста.

Модуль содержит реализацию стандартного обработчика эмбеддингов текста,
который используется по умолчанию.
"""

from typing import List, Optional, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from novel_analyser.core.config import get_config
from novel_analyser.core.interfaces import BaseEmbeddingEncoder
from novel_analyser.utils.logging import get_logger

logger = get_logger(__name__)


class StandardEmbeddingProcessor(BaseEmbeddingEncoder):
    """
    Стандартная реализация обработчика эмбеддингов текста.

    Использует модели sentence-transformers для кодирования текста в эмбеддинги.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        Инициализирует стандартный обработчик эмбеддингов.

        Args:
            model_name: Название модели для использования
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        super().__init__()

        # Получаем конфигурацию
        config = get_config()

        # Определяем имя модели и устройство
        self.model_name = model_name or config.model.embedding_model
        self.device = device or (
            "cuda"
            if config.model.use_gpu and torch.cuda.is_available()
            else "cpu"
        )

        # Добавляем кэш для часто используемых текстов
        self._cache: Dict[str, np.ndarray] = {}

        # Инициализируем модель
        logger.info(f"Загрузка модели для эмбеддингов: {self.model_name}")
        try:
            self.model = SentenceTransformer(
                self.model_name, device=self.device
            )
            logger.info(
                f"Используемое устройство для вычислений: {self.device}"
            )
            logger.info(
                f"Размерность эмбеддингов: {self.get_embedding_dimension()}"
            )
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def encode(
        self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги с использованием кэширования для повторяющихся текстов.

        Args:
            texts: Список текстов для кодирования
            show_progress: Показывать ли прогресс кодирования

        Returns:
            Массив эмбеддингов размерности (len(texts), embedding_dim)
        """
        if not texts:
            logger.warning("Получен пустой список текстов для кодирования")
            return np.zeros((0, self.get_embedding_dimension()))

        # Разделяем тексты на те, что есть в кэше, и те, что нужно закодировать
        texts_to_encode = []
        text_indices = []
        cached_embeddings = {}

        for i, text in enumerate(texts):
            if not text or text.isspace():
                cached_embeddings[i] = np.zeros(self.get_embedding_dimension())
                continue

            if text in self._cache:
                cached_embeddings[i] = self._cache[text]
            else:
                texts_to_encode.append(text)
                text_indices.append(i)

        # Если все уже в кэше, просто возвращаем результат
        if not texts_to_encode:
            result = np.zeros((len(texts), self.get_embedding_dimension()))
            for i, embedding in cached_embeddings.items():
                result[i] = embedding
            return result

        try:
            new_embeddings = self.model.encode(
                texts_to_encode, show_progress_bar=show_progress
            )
            result = np.zeros((len(texts), self.get_embedding_dimension()))

            for i, embedding in cached_embeddings.items():
                result[i] = embedding

            for i, (idx, text) in enumerate(
                zip(text_indices, texts_to_encode)
            ):
                embedding = new_embeddings[i]
                result[idx] = embedding

                # Кэшируем только непустые тексты
                if text and text.isspace():
                    self._cache[text] = embedding

            return result
        except Exception as e:
            logger.error(f"Ошибка при кодировании текстов: {str(e)}")
            return np.zeros((len(texts), self.get_embedding_dimension()))

    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.

        Returns:
            Размерность эмбеддинга
        """
        return self.model.get_sentence_embedding_dimension()

    def clear_cache(self) -> None:
        """Очищает кэш эмбеддингов."""
        self._cache.clear()
