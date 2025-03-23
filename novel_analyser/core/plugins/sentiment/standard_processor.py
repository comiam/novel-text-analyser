"""
Стандартная реализация обработчика эмоциональной окраски текста.

Модуль содержит реализацию стандартного обработчика эмоциональной окраски текста,
который используется по умолчанию.
"""

from typing import List, Literal, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
)

from novel_analyser.core.config import get_config
from novel_analyser.core.interfaces import BaseSentimentProcessor
from novel_analyser.utils.logging import get_logger

logger = get_logger(__name__)


class StandardSentimentProcessor(BaseSentimentProcessor):
    """
    Стандартная реализация обработчика эмоциональной окраски текста.

    Использует модели трансформеров для анализа эмоциональной окраски текста.
    """

    def __init__(
        self,
        split_chunk_max_length: int = 512,
        split_chunk_overlap: int = 50,
        **kwargs,
    ):
        """
        Инициализирует стандартный обработчик эмоциональной окраски.

        Args:
            split_chunk_max_length: Максимальная длина фрагмента текста
            split_chunk_overlap: Перекрытие между фрагментами текста
        """
        super().__init__()

        # Сохраняем переданные аргументы
        self.split_chunk_max_length = split_chunk_max_length
        self.split_chunk_overlap = split_chunk_overlap

        # Получаем конфигурацию
        config = get_config()

        # Определение устройства
        self.device = (
            "cuda"
            if torch.cuda.is_available() and config.model.use_gpu
            else "cpu"
        )
        logger.info(f"Используемое устройство для вычислений: {self.device}")

        # Инициализируем модель и токенизатор
        self.model_name = config.model.sentiment_model
        logger.info(
            f"Загрузка модели для анализа настроений: {self.model_name}"
        )

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            config=self.model_config,
            torch_dtype=(torch.float16 if self.device == "cuda" else None),
        )

        # Перемещение модели на устройство
        self.model = self.model.to(self.device)

        # Вычисляем безопасную максимальную длину для модели
        model_max_length = getattr(self.tokenizer, "model_max_length", 512)
        self.safe_max_length = min(model_max_length, 512)
        logger.info(
            f"Установлена максимальная длина последовательности в модель: {self.safe_max_length} токенов"
        )

    def get_sentiment(self, text: str) -> float:
        """
        Вычисляет эмоциональную оценку текста.

        Args:
            text: Текст для анализа

        Returns:
            Оценка настроения (от -1 до 1)
        """
        if not text or text.isspace():
            logger.debug(
                "Получен пустой текст для анализа настроения, возвращаю 0.0"
            )
            return 0.0

        try:
            with torch.no_grad():
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.safe_max_length,
                    return_tensors="pt",
                    padding="max_length",
                )

                # Перемещение на устройство
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                outputs = self.model(**encoding).logits
                proba = torch.sigmoid(outputs).cpu().numpy()[0]

            result = float(proba.dot([-1, 0, 1]))
            logger.debug(f"Вычислена эмоциональная оценка: {result}")
            return result
        except Exception as e:
            logger.error(f"Критическая ошибка обработки текста: {str(e)}")
            return 0.0

    def split_text_into_chunks(
        self, text: str, max_length: int = None, overlap: int = None
    ) -> List[str]:
        """
        Разделяет длинный текст на перекрывающиеся фрагменты.

        Args:
            text: Исходный текст для разделения
            max_length: Максимальная длина фрагмента в токенах
            overlap: Количество перекрывающихся токенов между фрагментами

        Returns:
            Список фрагментов текста
        """
        if not text or text.isspace():
            logger.debug(
                "Получен пустой текст для разделения на фрагменты, возвращаю пустой список"
            )
            return []

        logger.debug(
            f"Разделение текста длиной {len(text)} символов на фрагменты"
        )

        # Используем значения по умолчанию, если не указаны
        max_length = max_length or self.safe_max_length
        overlap = overlap or self.split_chunk_overlap

        # Разделение по предложениям для более естественных чанков
        sentences = [
            s.strip() for s in text.replace("\n", " ").split(".") if s.strip()
        ]

        if not sentences:
            return self._split_by_words(text, max_length)

        logger.debug(f"Обнаружено {len(sentences)} предложений")
        return self._split_by_sentences(sentences, max_length)

    def _split_by_words(self, text: str, max_length: int) -> List[str]:
        """
        Разделяет текст на фрагменты по словам.

        Args:
            text: Исходный текст
            max_length: Максимальная длина фрагмента

        Returns:
            Список фрагментов текста
        """
        words = text.split()
        logger.debug(
            f"Предложения не обнаружены, разбиваю текст на {len(words)} слов"
        )

        if not words:
            return []

        # Группируем слова в чанки
        chunks = []
        current_chunk = []
        current_length = 0
        max_words = int(max_length * 0.9)  # Примерная оценка

        for word in words:
            if current_length + 1 > max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = 1
            else:
                current_chunk.append(word)
                current_length += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.info(f"Создано {len(chunks)} фрагментов текста")
        return chunks

    def _split_by_sentences(
        self, sentences: List[str], max_length: int
    ) -> List[str]:
        """
        Разделяет текст на фрагменты по предложениям.

        Args:
            sentences: Список предложений
            max_length: Максимальная длина фрагмента

        Returns:
            Список фрагментов текста
        """
        # Собираем предложения в чанки
        chunks = []
        current_chunk = []
        current_est_tokens = 0
        max_est_tokens = int(max_length * 0.9)  # Консервативная оценка

        for sentence in sentences:
            # Грубая оценка количества токенов в предложении
            est_tokens = len(sentence) / 3  # ~3 символа на токен для русского

            if current_est_tokens + est_tokens > max_est_tokens:
                if current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [sentence]
                current_est_tokens = est_tokens
            else:
                current_chunk.append(sentence)
                current_est_tokens += est_tokens

        # Добавляем последний чанк
        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        logger.debug(f"Создано {len(chunks)} фрагментов текста")
        return chunks

    def _calculate_chunk_weight(
        self,
        i: int,
        total_chunks: int,
        chunk_length: int,
        total_length: int,
        sentiment: float,
        weighting_strategy: str,
    ) -> Tuple[float, float]:
        """
        Вычисляет вес фрагмента текста для анализа настроений.

        Args:
            i: Индекс фрагмента
            total_chunks: Общее количество фрагментов
            chunk_length: Длина текущего фрагмента
            total_length: Общая длина текста
            sentiment: Значение настроения фрагмента
            weighting_strategy: Стратегия взвешивания

        Returns:
            Кортеж (sentiment, weight) - настроение и вес фрагмента
        """
        # Абсолютное значение сентимента как мера уверенности модели
        confidence = abs(sentiment)

        # Вес по длине: более длинные фрагменты имеют больший вес
        length_weight = (
            chunk_length / total_length if total_length > 0 else 1.0
        )

        # Вес по позиции: учитывает положение фрагмента в тексте
        normalized_position = (
            i / max(1, total_chunks - 1) if total_chunks > 1 else 0.5
        )

        # Выбор стратегии позиционного взвешивания
        if weighting_strategy == "narrative":
            # Параболическая функция с пиком в начале и конце текста
            position_factor = 0.8 + 0.2 * (
                2 * (normalized_position - 0.5) ** 2
            )
        elif weighting_strategy == "speech":
            # Линейно убывающая функция
            position_factor = 1.0 - 0.2 * normalized_position
        else:  # "equal"
            # Равномерный вес для всех позиций
            position_factor = 1.0

        # Вес по уверенности: учитывает силу эмоционального окраса
        confidence_factor = 0.3 + 0.7 * min(1.0, confidence)

        # Финальный вес - произведение всех факторов
        weight = length_weight * position_factor * confidence_factor

        return sentiment, weight

    def analyze_long_text(
        self,
        text: str,
        weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> float:
        """
        Анализирует длинный текст, разбивая его на фрагменты и объединяя результаты.

        Args:
            text: Текст для анализа
            weighting_strategy: Стратегия взвешивания фрагментов:
                - "equal" - равный вес для всех позиций (по умолчанию)
                - "narrative" - больший вес началу и концу (для художественных текстов)
                - "speech" - больший вес началу (для человеческой речи)

        Returns:
            Итоговая оценка настроения
        """
        if not text or text.isspace():
            logger.debug("Получен пустой текст для анализа, возвращаю 0.0")
            return 0.0

        logger.debug(
            f"Анализ длинного текста, стратегия взвешивания: {weighting_strategy}"
        )

        # Используем более безопасный метод разделения текста
        chunks = self.split_text_into_chunks(text)

        if not chunks:
            logger.debug(
                "После разделения не получено фрагментов текста, возвращаю 0.0"
            )
            return 0.0

        if len(chunks) == 1:
            logger.debug(
                "Получен один фрагмент текста, выполняю прямой анализ"
            )
            return self.get_sentiment(chunks[0])

        # Оцениваем длину каждого чанка для весов
        chunk_lengths = [len(chunk) for chunk in chunks]
        total_length = sum(chunk_lengths)

        # Обрабатываем каждый фрагмент и собираем результаты
        weighted_sentiments = []
        total_weight = 0.0

        for i, chunk in enumerate(chunks):
            try:
                if not chunk or chunk.isspace():
                    continue

                # Получаем значение сентимента для текущего фрагмента
                sentiment = self.get_sentiment(chunk)

                # Рассчитываем вес для текущего фрагмента
                sentiment, weight = self._calculate_chunk_weight(
                    i,
                    len(chunks),
                    chunk_lengths[i],
                    total_length,
                    sentiment,
                    weighting_strategy,
                )

                weighted_sentiments.append((sentiment, weight))
                total_weight += weight

                logger.debug(
                    f"Фрагмент {i + 1}/{len(chunks)}: сентимент={sentiment:.3f}, вес={weight:.3f}"
                )
            except Exception as e:
                logger.error(f"Ошибка при анализе фрагмента {i + 1}: {str(e)}")
                continue

        # Вычисляем взвешенное среднее сентиментов
        if not weighted_sentiments:
            logger.warning(
                "Не удалось проанализировать ни один фрагмент, возвращаю 0.0"
            )
            return 0.0

        weighted_sum = sum(s * w for s, w in weighted_sentiments)
        result = weighted_sum / total_weight if total_weight else 0.0

        logger.debug(f"Итоговая взвешенная оценка настроения: {result:.3f}")
        return result
