"""Modulo de avaliacao do EASSE-PT."""

from easse_pt.metrics.bleu import corpus_bleu, sentence_bleu
from easse_pt.metrics.quality import corpus_features, sentence_features
from easse_pt.metrics.readability import (
    corpus_flesch_brasil,
    flesch_brasil,
    flesch_brasil_level,
    text_statistics,
)
from easse_pt.metrics.sari import corpus_sari, sentence_sari
from easse_pt.metrics.semantic import bertscore_pt, meaning_preservation, sbert_similarity

__all__ = [
    "corpus_bleu", "sentence_bleu",
    "corpus_sari", "sentence_sari",
    "flesch_brasil", "corpus_flesch_brasil", "flesch_brasil_level", "text_statistics",
    "bertscore_pt", "sbert_similarity", "meaning_preservation",
    "corpus_features", "sentence_features",
]
