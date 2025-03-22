"""
Различные анализаторы для обработки текста.
"""

from novel_analyser.analysers.basic import BasicAnalyser
from novel_analyser.analysers.character import CharacterAnalyser
from novel_analyser.analysers.clustering import ClusteringAnalyser
from novel_analyser.analysers.narrative import NarrativeAnalyser
from novel_analyser.analysers.readability import ReadabilityAnalyser
from novel_analyser.analysers.repetition import RepetitionAnalyser
from novel_analyser.analysers.semantic import SemanticAnalyser
from novel_analyser.analysers.sentiment import SentimentAnalyser
from novel_analyser.analysers.topic import TopicAnalyser

__all__ = [
    "BasicAnalyser",
    "ReadabilityAnalyser",
    "NarrativeAnalyser",
    "SentimentAnalyser",
    "TopicAnalyser",
    "CharacterAnalyser",
    "RepetitionAnalyser",
    "ClusteringAnalyser",
    "SemanticAnalyser",
]
