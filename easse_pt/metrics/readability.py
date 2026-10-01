"""Legibilidade em Portugues do Brasil — indice Flesch-Brasil.

Substitui o FKGL do EASSE, cuja calibracao (e cuja contagem de silabas) e
especifica do ingles. Usa-se a adaptacao de Martins et al. (1996) da formula
de Flesch para o Portugues do Brasil:

    ILF = 248.835 - (1.015 x ASL) - (84.6 x ASW)

    ASL = palavras / sentencas        (comprimento medio de sentenca)
    ASW = silabas / palavras          (comprimento medio de palavra, em silabas)

Faixas de interpretacao usuais (Martins et al., 1996):

    75-100  muito facil    (1a a 4a serie)
     50-75  facil          (5a a 8a serie)
     25-50  dificil        (ensino medio / superior)
      0-25  muito dificil  (academico)

A contagem de silabas usa Pyphen (dicionario pt_BR, baseado em hifenizacao) e,
se indisponivel, um contador heuristico de grupos vocalicos com tratamento de
ditongos, hiatos e encontros consonantais tipicos do PT.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Dict, List, Sequence

from easse_pt.tokenization import sentence_tokenize, word_tokenize

_VOWELS = set("aeiouáéíóúâêôàãõäëïöüy")
_WORD_ONLY = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$", re.UNICODE)

# Ditongos e tritongos do PT que contam como uma unica silaba.
_DIPHTHONGS = (
    "iai", "uai", "uei", "uou", "uiu",
    "ai", "ei", "oi", "ui", "au", "eu", "iu", "ou",
    "ãe", "ão", "õe", "ãi",
    "ia", "ie", "io", "iu", "ua", "ue", "uo",
)


@lru_cache(maxsize=1)
def _pyphen():
    try:
        import pyphen  # type: ignore

        for lang in ("pt_BR", "pt_PT", "pt"):
            if lang in pyphen.LANGUAGES:
                # left/right=1: em portugues a translineacao segue a divisao silabica,
                # mas os dicionarios nao separam silabas de 1 letra nas pontas
                # por padrao ("uni-ver-si-da-de" = 5, em vez de 6 silabas).
                return pyphen.Pyphen(lang=lang, left=1, right=1)
    except Exception:
        pass
    return None


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def count_syllables(word: str) -> int:
    """Numero de silabas de uma palavra em portugues (>= 1 para palavras)."""
    w = word.lower().strip("-'")
    if not w or not _WORD_ONLY.match(w):
        return 0

    dic = _pyphen()
    if dic is not None:
        n = len(dic.inserted(w).split("-"))
        return max(1, n)
    return _syllables_heuristic(w)


def _syllables_heuristic(w: str) -> int:
    i = 0
    count = 0
    n = len(w)
    while i < n:
        ch = w[i]
        if ch in _VOWELS:
            # tenta casar o maior ditongo/tritongo comecando aqui
            matched = 0
            for d in _DIPHTHONGS:
                if w.startswith(d, i) and len(d) > matched:
                    matched = len(d)
            count += 1
            i += matched if matched else 1
        else:
            i += 1
    # "qu"/"gu" antes de e/i nao formam silaba propria
    count -= len(re.findall(r"[qg]u[ei]", w))
    return max(1, count)


def text_statistics(text: str, tokenizer_backend: str = "auto") -> Dict[str, float]:
    """Contagens basicas de um texto: sentencas, palavras, silabas."""
    sents = sentence_tokenize(text, backend=tokenizer_backend) or [text]
    words: List[str] = []
    for s in sents:
        words.extend(t for t in word_tokenize(s, backend=tokenizer_backend) if _WORD_ONLY.match(t))
    syllables = sum(count_syllables(w) for w in words)
    return {
        "n_sentences": float(len(sents)),
        "n_words": float(len(words)),
        "n_syllables": float(syllables),
        "asl": len(words) / len(sents) if sents else 0.0,
        "asw": syllables / len(words) if words else 0.0,
    }


def flesch_brasil(text: str, tokenizer_backend: str = "auto") -> float:
    """Indice Flesch adaptado ao PT-BR (Martins et al., 1996). Maior = mais facil."""
    st = text_statistics(text, tokenizer_backend=tokenizer_backend)
    if st["n_words"] == 0:
        return 0.0
    return 248.835 - 1.015 * st["asl"] - 84.6 * st["asw"]


def flesch_brasil_level(score: float) -> str:
    """Rotulo textual da faixa de dificuldade."""
    if score >= 75:
        return "muito facil"
    if score >= 50:
        return "facil"
    if score >= 25:
        return "dificil"
    return "muito dificil"


def corpus_flesch_brasil(sentences: Sequence[str], tokenizer_backend: str = "auto") -> float:
    """Flesch-Brasil do corpus, calculado sobre as contagens agregadas.

    Agregar contagens (e nao promediar notas por sentenca) evita que sentencas
    muito curtas dominem a media — mesma pratica do FKGL de corpus no EASSE.
    """
    n_sents = n_words = n_syll = 0
    for text in sentences:
        st = text_statistics(text, tokenizer_backend=tokenizer_backend)
        n_sents += int(st["n_sentences"])
        n_words += int(st["n_words"])
        n_syll += int(st["n_syllables"])
    if n_words == 0 or n_sents == 0:
        return 0.0
    return 248.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syll / n_words)
