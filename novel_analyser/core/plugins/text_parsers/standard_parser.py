"""
Стандартная реализация парсера текста.

Модуль содержит реализацию стандартного парсера текста,
который используется по умолчанию.
"""

import re
from typing import Dict, List, Set

import numpy as np
import pymorphy3
from pydantic import BaseModel, Field
from razdel import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from novel_analyser.core.config import get_config
from novel_analyser.core.interfaces.parser import BaseParser
from novel_analyser.utils.stopwords import get_stop_words

IGNORE_BLOCK_NAMES: Set[str] = {
    "StoryData",
    "StoryTitle",
    "StoryStylesheet",
    "StoryScript",
}


class Block(BaseModel):
    """
    Представляет блок текста из Harlowe-формата.

    Attributes:
        name (str): Заголовок блока
        lines (List[str]): Строки содержимого блока
    """

    name: str = Field(..., description="Заголовок блока")
    lines: List[str] = Field(
        default_factory=list, description="Строки содержимого блока"
    )


class StandardParser(BaseParser):
    """
    Стандартная реализация парсера текста.

    Реализует методы базового класса BaseParser для разбора текста
    в формате Harlowe и извлечения диалогов персонажей.
    """

    def __init__(
            self,
            character_name_uppercase: bool = True,
            ignore_html_comments: bool = True,
            exclude_frequent_words_count: int = 30,
            **kwargs
    ):
        """
        Инициализирует стандартный парсер.

        Args:
            character_name_uppercase: Считать строку в верхнем регистре именем персонажа
            ignore_html_comments: Игнорировать HTML-комментарии
            exclude_frequent_words_count: Количество наиболее частых слов для исключения
        """
        super().__init__()
        self.morph = pymorphy3.MorphAnalyzer()
        self.character_name_uppercase = character_name_uppercase
        self.ignore_html_comments = ignore_html_comments
        self.exclude_frequent_words_count = exclude_frequent_words_count

    def parse_blocks(self, text: str, raw_style: bool = False) -> List[str]:
        """
        Разбирает текст на блоки, очищает их и возвращает список отфильтрованных блоков текста.

        Args:
            text: Исходный текст для разбора.
            raw_style: Флаг для сохранения блоков в сыром виде. По умолчанию False.

        Returns:
            Список отфильтрованных и очищенных блоков текста.
        """
        blocks_dict: Dict[str, Block] = self._parse_harlowe(text)
        # Если запрошен "сырый" стиль — возвращаем сырые блоки без чистки
        if raw_style:
            raw_blocks: List[str] = []
            for block in blocks_dict.values():
                if not block:
                    continue
                if any(
                        ignore_name in block.name
                        for ignore_name in IGNORE_BLOCK_NAMES
                ):
                    continue
                raw_text: str = "\n".join(block.lines)
                if raw_text:
                    raw_blocks.append(raw_text)
            return raw_blocks

        # Иначе выполняется стандартная обработка блоков
        stop_words: Set[str] = set(get_stop_words())
        morph: pymorphy3.MorphAnalyzer = self.morph

        cleaned_blocks: List[str] = []
        for block in blocks_dict.values():
            if not block:
                continue

            block_text: str = self._clean_block_text(block)
            if not block_text:
                continue

            lemmas: List[str] = self._lemmatize_and_tokenize_words(
                stop_words, morph, block_text
            )
            if lemmas:
                cleaned_blocks.append(" ".join(lemmas))

        filtered_blocks: List[str] = self._exclude_most_frequent_words(
            cleaned_blocks
        )
        return filtered_blocks

    def parse_character_dialogues(
            self, blocks: List[str]
    ) -> Dict[str, List[str]]:
        """
        Парсит блоки текста и извлекает реплики персонажей.

        Структура диалогов:
        ```
        ГЕРОЙ 1
        Реплика героя 1.
        Всё ещё реплика героя 1.

        ГЕРОЙ 2
        Реплика героя 2.
        ```

        Args:
            blocks (List[str]): Список текстовых блоков для анализа.

        Returns:
            Dict[str, List[str]]: Словарь, где ключи - имена персонажей, а значения - списки их реплик.
        """
        character_dialogues = {}
        current_character = None

        for block in blocks:
            lines = block.strip().split("\n")
            i = 0

            while i < len(lines):
                line = lines[i].strip()

                # Если строка полностью в верхнем регистре, считаем её потенциальным именем персонажа
                # Игнорируем HTML-комментарии и строки с #
                if (
                        line
                        and line.isupper()
                        and not line.startswith("#")
                        and not "<!--" in line
                ):
                    potential_character = line
                    # Удаляем комментарии, если они есть
                    if "#" in potential_character:
                        potential_character = potential_character.split("#")[
                            0
                        ].strip()

                    # Собираем реплику персонажа (может быть на нескольких строках)
                    dialogue = []
                    j = i + 1
                    while (
                            j < len(lines)
                            and lines[j].strip()
                            and not lines[j].strip().isupper()
                    ):
                        dialogue_line = lines[j].strip()
                        # Удаляем HTML-комментарии
                        if "<!--" in dialogue_line:
                            dialogue_line = re.sub(
                                r"<!--.*?-->", "", dialogue_line
                            )
                        # Удаляем начальный дефис если он есть
                        if dialogue_line.startswith("-"):
                            dialogue_line = dialogue_line[1:].strip()

                        dialogue.append(dialogue_line)
                        j += 1

                    # Только если есть диалог после имени персонажа, добавляем его
                    if dialogue:  # Если реплика не пустая
                        current_character = potential_character
                        if current_character not in character_dialogues:
                            character_dialogues[current_character] = []

                        character_dialogues[current_character].append(
                            " ".join(dialogue)
                        )
                        i = j  # Перепрыгиваем обработанные строки диалога
                    else:
                        i += 1  # Если диалога нет, просто переходим к следующей строке
                else:
                    i += 1  # Пропускаем нарративные строки или пустые строки

        return character_dialogues

    def extract_sentences(self, text: str) -> List[str]:
        """
        Извлекает предложения из текста.

        Args:
            text: Текст для извлечения предложений.

        Returns:
            Список предложений.
        """
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    def _parse_harlowe(self, content: str) -> Dict[str, Block]:
        """
        Разбивает исходный Harlowe‑файл на блоки.
        Блоки идентифицируются строками, начинающимися с "::".

        Args:
            content (str): Содержимое Harlowe-файла

        Returns:
            Dict[str, Block]: Словарь блоков {имя_блока: объект_блока}
        """
        blocks: Dict[str, Block] = {}
        current_block_name: str = ""
        current_block_lines: List[str] = []
        ignore_names: Set[str] = IGNORE_BLOCK_NAMES

        for line in content.splitlines():
            if line.startswith("::"):
                if (
                        current_block_name
                        and current_block_name not in ignore_names
                ):
                    blocks[current_block_name] = Block(
                        name=current_block_name, lines=current_block_lines
                    )
                header: str = self._clean_block_header(line[2:].strip())
                current_block_name = header
                current_block_lines = []
            else:
                if current_block_name:
                    current_block_lines.append(line)

        if current_block_name and current_block_name not in ignore_names:
            blocks[current_block_name] = Block(
                name=current_block_name, lines=current_block_lines
            )
        return blocks

    def _clean_block_header(self, header: str) -> str:
        """
        Очищает заголовок блока, удаляя лишние данные:
          - Отсекает всё после обратного слэша,
          - Удаляет содержимое в квадратных и фигурных скобках,
          - Убирает ведущие дефисы.

        Args:
            header (str): Исходный заголовок блока

        Returns:
            str: Очищенный заголовок
        """
        header = header.split("\\", 1)[0].strip()
        header = re.sub(r"\s*\[.*$", "", header).strip()
        header = re.sub(r"\s*\{.*$", "", header).strip()
        return header.lstrip("-").strip()

    def _clean_block_text(self, block: Block) -> str:
        """
        Очищает текст блока от нежелательных символов и слов.
        Если в имени блока содержится "StoryStylesheet", "StoryScript",
        "StoryData" или "StoryTitle", возвращается пустая строка.

        Args:
            block: Блок текста, который нужно очистить.

        Returns:
            Очищенный текст блока.
        """
        block_text: str = "\n".join(block.lines)

        if any(
                ignore_name in block.name for ignore_name in IGNORE_BLOCK_NAMES
        ):
            return ""

        regex = r"[=\(\)\-\!\"№\;%\:\?\$\*\+\-@0-9]|[a-zA-Z]|\b[А-Я]+\b"
        block_text = re.sub(regex, "", block_text)

        return block_text

    def _lemmatize_and_tokenize_words(
            self,
            stop_words: Set[str],
            morph: pymorphy3.MorphAnalyzer,
            block_text: str,
    ) -> List[str]:
        """
        Лемматизирует и токенизирует слова в заданном тексте, исключая стоп-слова.

        Args:
            stop_words: Набор стоп-слов, которые нужно исключить.
            morph: Экземпляр морфологического анализатора pymorphy3.
            block_text: Текстовый блок для обработки.

        Returns:
            Список лемматизированных и токенизированных слов, исключая стоп-слова.
        """
        tokens: List[str] = [
            token.text.lower() for token in tokenize(block_text)
        ]

        lemmas: List[str] = [
            morph.parse(token)[0].normal_form
            for token in tokens
            if token
               and token not in stop_words
               or morph.parse(token)[0].normal_form not in stop_words
        ]

        return lemmas

    def _get_names_words(self) -> List[str]:
        """
        Возвращает список имен персонажей.

        Returns:
            Список имен персонажей.
        """
        # Initialize with an empty list in case the config's predefined_names is not available
        try:
            config = get_config()
            return config.character.predefined_names
        except AttributeError:
            # Handle the case where the config structure doesn't match what we expect
            # This could happen during transition to the new Pydantic structure
            print(
                "Warning: Could not access config.character.predefined_names, returning empty list"
            )
            return []

    def _exclude_most_frequent_words(
            self, cleaned_blocks: List[str]
    ) -> List[str]:
        """
        Исключает наиболее часто встречающиеся слова из списка текстовых блоков.

        Args:
            cleaned_blocks: Список текстовых блоков, из которых нужно исключить частые слова.

        Returns:
            Список текстовых блоков с исключенными частыми словами.
        """
        vectorizer: TfidfVectorizer = TfidfVectorizer(max_df=1.0)
        tfidf_matrix = vectorizer.fit_transform(cleaned_blocks)
        feature_names = vectorizer.get_feature_names_out()

        tfidf_sum: np.ndarray = np.array(tfidf_matrix.sum(axis=0)).flatten()
        top_indices = tfidf_sum.argsort()[::-1]

        frequent_words: Set[str] = {
            feature_names[i]
            for i in top_indices[: self.exclude_frequent_words_count]
            if feature_names[i] not in self._get_names_words()
        }

        filtered_blocks: List[str] = []
        for block in cleaned_blocks:
            words: List[str] = block.split()

            filtered_block: str = " ".join(
                word for word in words if word not in frequent_words
            )

            filtered_blocks.append(filtered_block if filtered_block else block)

        return filtered_blocks
