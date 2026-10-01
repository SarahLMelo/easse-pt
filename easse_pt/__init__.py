"""EASSE-PT — Easier Automatic Sentence Simplification Evaluation para o Portugues.

Adaptacao do framework EASSE (Alva-Manchego et al., 2019) para PT-BR:
tokenizacao em portugues, SARI/BLEU sobre essa tokenizacao, indice
Flesch-Brasil no lugar do FKGL e um modulo de preservacao semantica
com embeddings em portugues.
"""

__version__ = "0.1.0"

from easse_pt.tokenization import tokenize, word_tokenize, sentence_tokenize  # noqa: F401

__all__ = ["__version__", "tokenize", "word_tokenize", "sentence_tokenize"]
