"""
Функции для работы с эмбеддингами текста.
"""

from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


class EmbeddingModelConfig(BaseModel):
    """Конфигурация модели эмбеддингов."""

    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Название модели для использования",
    )
    device: Optional[str] = Field(
        default=None,
        description="Устройство для вычислений ('cuda' или 'cpu')",
    )


class EmbeddingModel:
    """
    Класс для получения эмбеддингов текста.
    """

    def __init__(
            self,
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device: Optional[str] = None,
    ):
        """
        Инициализирует модель для получения эмбеддингов.

        Args:
            model_name: Название модели для использования
            device: Устройство для вычислений ('cuda' или 'cpu')
        """
        config = EmbeddingModelConfig(model_name=model_name, device=device)

        if config.device is None:
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = config.model_name
        self.device = config.device
        self.model = SentenceTransformer(
            config.model_name, device=config.device
        )

    def encode(
            self, texts: List[str], show_progress: bool = True
    ) -> np.ndarray:
        """
        Получает эмбеддинги для списка текстов.

        Args:
            texts: Список текстов для кодирования
            show_progress: Показывать ли прогресс

        Returns:
            Массив эмбеддингов
        """
        return self.model.encode(texts, show_progress_bar=show_progress)


def get_text_embeddings(
        texts: List[str],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: Optional[str] = None,
) -> np.ndarray:
    """
    Получает эмбеддинги текста с использованием указанной модели.

    Args:
        texts: Список текстов для получения эмбеддингов
        model_name: Название модели для использования
        device: Устройство для выполнения вычислений ("cuda" или "cpu")

    Returns:
        Массив эмбеддингов для каждого текста
    """
    # Применяем настройки модели через EmbeddingModelConfig
    config = EmbeddingModelConfig(model_name=model_name, device=device)
    model = EmbeddingModel(config.model_name, config.device)
    return model.encode(texts)
