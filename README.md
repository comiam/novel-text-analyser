# Novel Analyser

Библиотека для комплексного анализа русскоязычных текстов визуальных новелл.

## Описание

Novel Analyser - это библиотека для всестороннего анализа текстов визуальных новелл и других литературных произведений на русском языке. Библиотека предлагает широкий набор инструментов для анализа текста, включая:

- Базовый анализ текста (количество блоков, время чтения)
- Анализ читаемости и сложности текста
- Нарративный анализ (структура повествования, ритм)
- Анализ эмоциональной окраски текста
- Тематическое моделирование
- Анализ персонажей и их диалогов
- Анализ повторяемости и стиля
- Кластеризацию текста на основе семантических эмбеддингов
- Семантический анализ (когерентность текста)

## Поддерживаемые форматы

На данный момент поддерживается только простой Twine формат сюжета. Библиотека может анализировать текстовые файлы, созданные с использованием Twine, где сюжет представлен в виде узлов и связей между ними.

Пример поддерживаемого синтаксиса сюжета:
```bash
# Реплика первого героя
ГЕРОЙ 1
Всё это пустой трёп.
Кабачки прекрасны.
Ты просто не шаришь, дурачок.

# Разделение между репликами - пуская строка.
ГЕРОЙ 2
Я хотяб ногти не крашу.

# Это нарратив без привязки к герою. Просто голос диктора или описание.
Герой 1 один был шокирован парированием героя 2 и потому активировал замедление времени.
```

### Ограничения текущей реализации

- Формат файла должен быть `.twee`

### Расширяемость на другие форматы

Это более сложная операция, но возможная. Библиотека поддерживает расширение парсинга различных форматов сюжета. Про это ниже в разделе "Расширяемость".

## Установка

### Требования

- Python 3.10 или выше
- pip (менеджер пакетов Python)

### Установка из исходников

```bash
# Клонирование репозитория
git clone https://github.com/comiam/text_analysis.git
cd text_analysis

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

### Комплексный анализ текста

Ниже приведен пример полного анализа текста с сохранением результатов и исследованием различных аспектов:

```python
from novel_analyser import TextAnalyser, configure
import os

# Настройка конфигурации - один раз при запуске
configure({
    "output": {
        "output_dir": "analysis_results"
    },
    "model": {
        "use_gpu": True,  # Использовать GPU если доступен
        "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    },
    "analyse": {
        "reading_speed_wpm": 180.0  # Скорость чтения слов в минуту для подсчета времени
    },
    "sentiment_analyze": {
        "positive_threshold": 0.1,   # Порог для позитивной оценки
        "negative_threshold": -0.1   # Порог для негативной оценки
    },
    "character": {
        "predefined_names": ["АЛИСА", "БОРЯ", "ВИКА"]  # Предопределенные имена персонажей
    }
})

# Создание анализатора
analyser = TextAnalyser()

# Путь к файлу с текстом новеллы
file_path = "path/to/your/novel.txt"

# Определение интересующих типов анализа
# Можно выбрать только нужные типы анализа: 
# 'basic', 'readability', 'narrative', 'sentiment', 
# 'topic', 'character', 'repetition', 'clustering', 'semantic'
analyses = [
    "basic",           # Базовый анализ текста
    "readability",     # Анализ читаемости
    "narrative",       # Нарративный анализ
    "sentiment",       # Анализ эмоциональной окраски
    "character",       # Анализ персонажей
    "topic"            # Тематическое моделирование
]

# Выполнение полного анализа текста
result = analyser.analyse_file(file_path, analyses=analyses)

# Вывод итогового резюме
print("\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===\n")
print(result.summary)
```

### Анализ эмоциональной окраски текста:

```python
from novel_analyser import SentimentAnalyser # так же есть и остальные модули анализа по отдельности

# Создаем анализатор настроений
sentiment_analyser = SentimentAnalyser()

# Анализ текста
result = sentiment_analyser.analyse(["Ваш текст для анализа"])

# Доступ к результатам
print(f"Средняя эмоциональная окраска: {result.metrics['avg_sentiment']}")
print(f"Процент положительных блоков: {result.metrics['pos_ratio'] * 100:.1f}%")
print(f"Процент отрицательных блоков: {result.metrics['neg_ratio'] * 100:.1f}%")
```

### Работа с эмбеддингами текста:

```python
from novel_analyser import EmbeddingProcessor

# Создаем процессор эмбеддингов
embedding_encoder = EmbeddingProcessor()

# Получаем эмбеддинги для списка текстов
texts = ["Первый текст", "Второй текст", "Третий текст"]
embeddings = embedding_encoder.encode(texts)

# Вычисляем схожесть между двумя текстами
similarity = embedding_encoder.compute_similarity("Кошка спит", "Кот дремлет")
print(f"Семантическая схожесть: {similarity:.4f}")

# Вычисляем матрицу схожести для списка текстов
similarity_matrix = embedding_encoder.compute_batch_similarities(texts)
print("Матрица схожести:")
print(similarity_matrix)
```

### Тематическое моделирование:

```python
from novel_analyser import TopicModeler

# Создаем анализатор тем
topic_analyser = TopicModeler()

# Находим оптимальное количество тем и ключевые слова
optimal_topics, topic_keywords = topic_analyser.extract_topics("Ваш текст")

print(f"Оптимальное количество тем: {optimal_topics}")
for topic_id, keywords in topic_keywords.items():
    print(f"Тема {topic_id}: {', '.join(keywords)}")
```

### Анализ персонажей:

```python
from novel_analyser import CharacterAnalyser

# Создаем анализатор персонажей
character_analyser = CharacterAnalyser()

# Получаем метрики персонажей
character_metrics = character_analyser.compute_character_metrics({
    "герой": "Ну как с сосисками?",
    "герой 2": "5 минут, турецкий"
})

for char_name, metrics in character_metrics.items():
    print(f"Персонаж: {char_name}")
    print(f"Метрики: {metrics}")
```

### Настройка конфигурации

```python
from novel_analyser import configure

configure({
    "output": {
        "output_dir": "my_analysis"
    },
    "sentiment_analyze": {
        "positive_threshold": 0.1,
        "negative_threshold": -0.1
    },
    "model": {
        "use_gpu": True
    },
    "embedding": {
        "args": {
            "model_name": "sentence-transformers/LaBSE",
            "show_progress_bar": False
        }
    }
})
```

## Структура результатов анализа

Результат анализа (`AnalysisResult`) содержит:

- `metrics`: словарь с числовыми метриками и другими результатами анализа
- `figures`: словарь с путями к сохраненным визуализациям
- `summary`: текстовое резюме результатов анализа

## Конфигурация

Библиотека использует конфигурационный файл в формате YAML. Конфигурация по умолчанию находится в `configs/default_config.yaml`. Вы можете изменить параметры, используя функцию `configure()` или создав свой конфигурационный файл.

## Логирование

Novel Analyser поддерживает настраиваемое логирование, которое можно сконфигурировать под ваши потребности:

```python
from novel_analyser.utils.logging import configure_logging
import logging

# Настройка логирования из конфиг-файла
configure_logging("path/to/logging_config.yaml")

# Получение логгера для вашего модуля
logger = logging.getLogger("your_module_name")

# Использование логгера
...
```

Библиотека поставляется с конфигурационным файлом логгирования по умолчанию `configs/logging_config.yaml`.

## Расширяемость

Novel Analyser спроектирован с учетом возможности расширения. Вы можете создавать собственные компоненты парсинга и обработки сюжета для различных аспектов анализа, не меняя основной код библиотеки.

### Собственные парсеры текста (можно Renpy, можно Snoflake и другие)

Для создания собственного парсера текста необходимо реализовать интерфейс `BaseParser`:

```python
from novel_analyser.core.interfaces.parser import BaseParser
from typing import Dict, List

class MyCustomParser(BaseParser):
    """Мой собственный парсер текста в формате X."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Инициализация парсера с дополнительными параметрами
        
    def parse_blocks(self, text: str, raw_style: bool = False) -> List[str]:
        """
        Разбирает текст на блоки.
        """
        # Ваша реализация
        return blocks
        
    def parse_character_dialogues(self, blocks: List[str]) -> Dict[str, List[str]]:
        """
        Извлекает диалоги персонажей.
        """
        # Ваша реализация
        return dialogues
        
    def extract_sentences(self, text: str) -> List[str]:
        """
        Извлекает предложения из текста.
        """
        # Ваша реализация
        return sentences
```

### Собственные обработчики эмоциональной окраски

Если хотите добавить свою модель для анализа настроения текста, то велкам. Для создания собственного обработчика эмоциональной окраски текста необходимо реализовать интерфейс `BaseSentimentProcessor`:

```python
from novel_analyser.core.interfaces.sentiment import BaseSentimentProcessor
from typing import List, Literal

class MyCustomSentimentProcessor(BaseSentimentProcessor):
    """Мой собственный обработчик эмоциональной окраски текста."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Инициализация обработчика с дополнительными параметрами
        
    def get_sentiment(self, text: str) -> float:
        """
        Вычисляет эмоциональную оценку для текста.
        """
        # Ваша реализация
        return sentiment
        
    def analyze_long_text(
        self,
        text: str,
        weighting_strategy: Literal["equal", "narrative", "speech"] = "equal",
    ) -> float:
        """
        Анализирует длинный текст, разбивая его на фрагменты.
        """
        # Ваша реализация
        return overall_sentiment
        
    def split_text_into_chunks(
        self, text: str, max_length: int, overlap: int
    ) -> List[str]:
        """
        Разделяет длинный текст на перекрывающиеся фрагменты.
        """
        # Ваша реализация
        return chunks
```

### Собственные обработчики эмбеддингов

Для создания собственного обработчика эмбеддингов текста необходимо реализовать интерфейс `BaseEmbeddingEncoder`:

```python
from novel_analyser.core.interfaces.embedding import BaseEmbeddingEncoder
import numpy as np
from typing import List

class MyCustomEmbeddingProcessor(BaseEmbeddingEncoder):
    """Мой собственный обработчик эмбеддингов текста."""
    
    def __init__(self, **kwargs):
        super().__init__()
        # Инициализация обработчика с дополнительными параметрами
        
    def encode(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Кодирует тексты в эмбеддинги.
        """
        # Ваша реализация
        return embeddings
        
    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.
        """
        # Ваша реализация (если нужно переопределить метод базового класса)
        return dimension
```

### Подключение собственных плагинов

Для использования собственных плагинов нужно указать путь к их классам в конфигурации:

```python
from novel_analyser import configure

configure({
    "parser": {
        "module_path": "your_module.parsers.CustomParser",
        "args": {
            "custom_param": "value"
        }
    },
    "sentiment": {
        "module_path": "your_module.sentiment.CustomSentimentProcessor",
        "args": {
            "model_path": "/path/to/model"
        }
    },
    "embedding": {
        "module_path": "your_module.embedding.CustomEmbeddingProcessor",
    }
})
```

### Регистрация собственных компонентов

После создания собственного компонента его необходимо зарегистрировать в системе:

```python
from novel_analyser.core.plugins import get_parser_registry, get_sentiment_processor_registry
from my_module import MyCustomParser, MyCustomSentimentProcessor

# Регистрация собственного парсера
parser_registry = get_parser_registry()
parser_registry.register("MyCustomParser", MyCustomParser)

# Регистрация собственного обработчика эмоциональной окраски
sentiment_registry = get_sentiment_processor_registry()
sentiment_registry.register("MyCustomSentimentProcessor", MyCustomSentimentProcessor)

# Настройка конфигурации для использования ваших компонентов
from novel_analyser import configure

configure({
    "parser": {
        "module_path": "my_module.MyCustomParser",
        "args": {
            "your_custom_param": "value"
        }
    },
    "sentiment": {
        "module_path": "my_module.MyCustomSentimentProcessor",
        "args": {
            "your_custom_param": "value"
        }
    }
})
```

### Создание собственных анализаторов

Вы также можете создавать полностью новые анализаторы, наследуясь от базового класса `BaseAnalyser`:

```python
from novel_analyser.core.base_analyser import BaseAnalyser, AnalysisResult
from typing import List, Optional

class MyCustomAnalyser(BaseAnalyser):
    """Мой собственный анализатор для XYZ."""
    
    def __init__(self, config=None):
        super().__init__(config)
        # Дополнительная инициализация
        
    def analyse(self, blocks: List[str]) -> AnalysisResult:
        """
        Выполняет анализ текстовых блоков.
        
        Args:
            blocks: Список текстовых блоков для анализа
            
        Returns:
            Результат анализа
        """
        result = AnalysisResult()
        
        # Ваша логика анализа
        # ...
        
        # Заполняем метрики
        result.metrics.update({
            "my_metric_1": value1,
            "my_metric_2": value2,
        })
        
        # Заполняем пути к сохраненным изображениям
        result.figures.update({
            "my_plot": self.save_figure("my_custom_plot.png"),
        })
        
        # Заполняем текстовое резюме
        result.summary = "Мой анализ:\n"
        result.summary += f"  Метрика 1: {value1}\n"
        result.summary += f"  Метрика 2: {value2}\n"
        
        return result
```

## Примеры визуализаций

Библиотека генерирует различные визуализации:

- Гистограммы распределения времени чтения, эмоциональной окраски и др.
- Круговые диаграммы распределения эмоциональной окраски
- Графики для определения оптимального количества тем и кластеров
- Диаграммы рассеяния для визуализации кластеров и тем
- Столбчатые диаграммы для анализа персонажей

## Автор

Максим

## Содействие

Вклады в проект приветствуются! Если у вас есть предложения или вы нашли ошибку, пожалуйста, создайте issue или pull request.
