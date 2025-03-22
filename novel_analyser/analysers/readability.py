"""
Модуль анализа читаемости текста.
"""

from typing import Dict, List, Optional

from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from novel_analyser.core.config import AnalyserConfig
from novel_analyser.core.text_processor import TextProcessor


class ReadabilityAnalyser(BaseAnalyser):
    """
    Класс для анализа читаемости текста.
    """

    def __init__(self, config: Optional[AnalyserConfig] = None):
        """
        Инициализирует анализатор читаемости.

        Args:
            config: Конфигурация анализатора
        """
        super().__init__(config)
        self.text_processor = TextProcessor()

    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет анализ читаемости текстовых блоков.

        Args:
            blocks: Список текстовых блоков для анализа

        Returns:
            Результат анализа
        """
        result = AnalysisResult()

        # Объединяем все блоки в один текст для общего анализа
        overall_text = " ".join(blocks)

        # Вычисляем метрики читаемости для каждого блока
        readability_metrics_list = [
            self.compute_readability_metrics(block) for block in blocks
        ]

        # Вычисляем общие метрики
        total_sentences = sum(
            m["num_sentences"] for m in readability_metrics_list
        )
        total_words = sum(m["num_words"] for m in readability_metrics_list)

        avg_words_per_sentence = (
            total_words / total_sentences if total_sentences > 0 else 0
        )

        avg_word_length = (
            sum(
                m["avg_word_length"] * m["num_words"]
                for m in readability_metrics_list
            )
            / total_words
            if total_words > 0
            else 0
        )

        # Вычисляем лексические метрики
        lexical_metrics = self.analyze_lexical_complexity(overall_text)

        # Анализируем распределение частей речи
        pos_distribution = self.text_processor.analyze_pos_distribution(
            overall_text
        )

        # Определяем возрастную группу
        estimated_age = self.estimate_age_readability(overall_text)

        # Заполняем метрики
        result.metrics.update(
            {
                "total_sentences": total_sentences,
                "total_words": total_words,
                "avg_words_per_sentence": avg_words_per_sentence,
                "avg_word_length": avg_word_length,
                "type_token_ratio": lexical_metrics["type_token_ratio"],
                "readability_index": lexical_metrics["readability_index"],
                "estimated_age_group": estimated_age,
                "pos_distribution": pos_distribution,
            }
        )

        # Заполняем текстовое резюме
        result.summary = "Лингвистический анализ и оценка читабельности:\n"
        result.summary += f"  Общее число предложений: {total_sentences}\n"
        result.summary += f"  Общее число слов: {total_words}\n"
        result.summary += f"  Среднее число слов в предложении: {avg_words_per_sentence:.2f}\n"
        result.summary += (
            f"  Средняя длина слова: {avg_word_length:.2f} символов\n"
        )
        result.summary += (
            f"  Type-token ratio: {lexical_metrics['type_token_ratio']:.3f}\n"
        )
        result.summary += f"  Индекс удобочитаемости (Flesch Reading Ease): {lexical_metrics['readability_index']:.2f}\n"
        result.summary += (
            f"  Ориентировочная читаемость для: {estimated_age}\n"
        )
        result.summary += "Распределение частей речи:\n"

        for pos, count in pos_distribution.items():
            result.summary += f"  {pos}: {count}\n"

        return result

    def compute_readability_metrics(self, text: str) -> Dict[str, float]:
        """
        Вычисляет базовые метрики читабельности.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с числом предложений, слов, средним числом слов в предложении и средней длиной слова
        """
        sentences: List[str] = self.text_processor.extract_sentences(text)
        num_sentences: int = len(sentences)

        words: List[str] = self.text_processor.tokenize(text)
        num_words: int = len(words)

        avg_words_per_sentence: float = (
            num_words / num_sentences if num_sentences > 0 else 0
        )
        avg_word_length: float = (
            sum(len(word) for word in words) / num_words
            if num_words > 0
            else 0
        )

        return {
            "num_sentences": num_sentences,
            "num_words": num_words,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_word_length": avg_word_length,
        }

    def compute_readability_index(self, text: str) -> float:
        """
        Вычисляет индекс удобочитаемости по формуле Флеша.

        Args:
            text: Текст для анализа

        Returns:
            Индекс удобочитаемости
        """
        metrics: Dict[str, float] = self.compute_readability_metrics(text)

        words: List[str] = self.text_processor.tokenize(text)
        if not words:
            return 0

        syllable_count: int = sum(
            self.text_processor.count_syllables(word) for word in words
        )
        avg_syllables: float = syllable_count / len(words)
        words_per_sentence: float = metrics["avg_words_per_sentence"]

        return 206.835 - 84.6 * avg_syllables - 1.015 * words_per_sentence

    def analyze_lexical_complexity(self, text: str) -> Dict[str, float]:
        """
        Анализ лексической сложности текста.

        Args:
            text: Текст для анализа

        Returns:
            Словарь с метриками: общее число слов, число уникальных слов, type-token ratio,
            индекс удобочитаемости и средняя длина предложения
        """
        words: List[str] = self.text_processor.tokenize(text)
        total_words: int = len(words)

        unique_words: int = len(set(words))
        type_token_ratio: float = (
            unique_words / total_words if total_words > 0 else 0
        )

        readability_index: float = self.compute_readability_index(text)
        metrics: Dict[str, float] = self.compute_readability_metrics(text)
        avg_sentence_length: float = metrics["avg_words_per_sentence"]

        return {
            "total_words": total_words,
            "unique_words": unique_words,
            "type_token_ratio": type_token_ratio,
            "readability_index": readability_index,
            "avg_sentence_length": avg_sentence_length,
        }

    def estimate_age_readability(self, text: str) -> str:
        """
        Рассчитывает ориентировочную читаемость текста для разных возрастных групп.

        Args:
            text: Текст для анализа

        Returns:
            Оценка читаемости (возрастная группа)
        """
        # Анализ лексической сложности текста
        lexical: Dict[str, float] = self.analyze_lexical_complexity(text)

        # Вычисление базовых метрик читабельности текста
        metrics: Dict[str, float] = self.compute_readability_metrics(text)

        # Анализ распределения частей речи
        pos_stats = self.text_processor.analyze_pos_distribution(text)

        # Получение количества глаголов в тексте
        num_verbs = pos_stats.get("VERB", 0)

        # Получение количества прилагательных (полное и краткое)
        num_adjs = pos_stats.get("ADJF", 0) + pos_stats.get("ADJS", 0)

        words: List[str] = self.text_processor.tokenize(text)
        if not words:
            return "Не определено"

        # Формирование списка слов, длиннее 7 символов
        long_words: List[str] = [word for word in words if len(word) > 7]
        ratio_long: float = len(long_words) / len(words)

        # Подсчёт слов с более чем 2 слогами
        polysyllabic_count: int = sum(
            self.text_processor.count_syllables(w) > 2 for w in words
        )

        # Вычисление отношений
        polysyllabic_word_ratio: float = polysyllabic_count / len(words)
        adj_ratio: float = num_adjs / len(words)
        verb_ratio: float = num_verbs / len(words)

        # Базовый индекс, комбинирующий среднюю длину предложения и слова
        base_index: float = (
                metrics["avg_words_per_sentence"]
                * metrics["avg_word_length"]
                * (1 + ratio_long)
        )

        # Композитный индекс
        composite_index: float = (
                base_index * (1 + lexical["type_token_ratio"])
                + 0.3 * lexical["readability_index"]
                + 0.2 * verb_ratio
                + 0.1 * polysyllabic_word_ratio
                + 0.1 * adj_ratio
        )

        if composite_index < 40:
            return "Детский (7-10 лет)"
        elif composite_index < 60:
            return "Младший школьный (10-13 лет)"
        elif composite_index < 80:
            return "Подростковый (13-17 лет)"
        else:
            return "Взрослый (17+)"
