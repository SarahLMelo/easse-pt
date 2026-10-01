"""Tokenizacao para o Portugues do Brasil.

O EASSE original usa tokenizadores voltados ao ingles (13a/Moses do sacrebleu),
que tratam mal clise com hifen ("diz-se", "far-lhe-ia"), contracoes ("d'agua")
e abreviaturas comuns em PT. Aqui ha tres backends, escolhidos por disponibilidade:

    spacy  -> pt_core_news_sm (melhor qualidade linguistica)
    nltk   -> punkt em portugues
    regex  -> fallback puro-Python, sem dependencia externa (padrao garantido)

Todas as metricas do EASSE-PT passam por `tokenize()`, de modo que SARI, BLEU e
Flesch-Brasil compartilham exatamente a mesma segmentacao — requisito para que
os numeros sejam comparaveis entre execucoes e entre trabalhos.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional

BACKENDS = ("spacy", "nltk", "regex")

# Abreviaturas frequentes em textos jornalisticos brasileiros — evitam quebra
# indevida de sentenca no backend regex.
_ABBREV = {
    "sr", "sra", "srta", "dr", "dra", "prof", "profa", "exmo", "exma",
    "av", "r", "km", "n", "no", "num", "pag", "fl", "art", "inc", "par",
    "cia", "ltda", "etc", "ex", "obs", "ref", "cf", "seg", "min", "jr",
    "ago", "set", "out", "nov", "dez", "jan", "fev", "mar", "abr", "jun", "jul",
}

# Palavra: letras (com acento), digitos, hifen interno e apostrofo interno.
_WORD_RE = re.compile(
    r"""
    \d+(?:[.,]\d+)*(?:[%º°ª])?          # numeros, percentuais, ordinais
    | \w+(?:[-'’]\w+)*             # palavras, clise com hifen, apostrofo
    | [^\w\s]                           # pontuacao isolada
    """,
    re.UNICODE | re.VERBOSE,
)

_SENT_END_RE = re.compile(r"(?<=[.!?…])[\"'»”’\)\]]*\s+")

_active_backend: Optional[str] = None


# --------------------------------------------------------------------------- #
# Selecao de backend
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _load_spacy():
    import spacy  # type: ignore

    try:
        return spacy.load("pt_core_news_sm", disable=["parser", "ner", "lemmatizer"])
    except OSError:  # modelo nao baixado
        return None


@lru_cache(maxsize=1)
def _load_nltk() -> bool:
    try:
        import nltk  # type: ignore

        nltk.data.find("tokenizers/punkt")
        return True
    except Exception:
        try:
            import nltk  # type: ignore

            nltk.download("punkt", quiet=True)
            nltk.data.find("tokenizers/punkt")
            return True
        except Exception:
            return False


def available_backend(preferred: str = "auto") -> str:
    """Resolve qual backend sera usado. `preferred` in {'auto','spacy','nltk','regex'}."""
    global _active_backend
    if preferred != "auto":
        if preferred not in BACKENDS:
            raise ValueError(f"backend desconhecido: {preferred!r} (use {BACKENDS})")
        _active_backend = preferred
        return preferred

    if _active_backend is not None:
        return _active_backend

    try:
        if _load_spacy() is not None:
            _active_backend = "spacy"
            return _active_backend
    except Exception:
        pass
    try:
        if _load_nltk():
            _active_backend = "nltk"
            return _active_backend
    except Exception:
        pass
    _active_backend = "regex"
    return _active_backend


def backend_info() -> dict:
    """Metadados do tokenizador ativo — vao para o cabecalho do relatorio."""
    backend = available_backend()
    detail = {"spacy": "pt_core_news_sm", "nltk": "punkt/pt", "regex": "builtin"}[backend]
    return {"tokenizer_backend": backend, "tokenizer_model": detail}


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def word_tokenize(text: str, backend: str = "auto") -> List[str]:
    """Lista de tokens de uma sentenca em portugues."""
    text = normalize(text)
    chosen = available_backend(backend)

    if chosen == "spacy":
        nlp = _load_spacy()
        if nlp is not None:
            return [t.text for t in nlp(text) if not t.is_space]
    elif chosen == "nltk" and _load_nltk():
        from nltk.tokenize import word_tokenize as _nltk_wt  # type: ignore

        return _nltk_wt(text, language="portuguese")

    return _WORD_RE.findall(text)


def tokenize(text: str, lowercase: bool = True, backend: str = "auto") -> List[str]:
    """Tokenizacao canonica usada por SARI/BLEU do EASSE-PT.

    Por padrao aplica lowercase, como faz o EASSE original ao calcular SARI,
    para que diferencas de caixa nao contem como edicoes.
    """
    toks = word_tokenize(text, backend=backend)
    return [t.lower() for t in toks] if lowercase else toks


def sentence_tokenize(text: str, backend: str = "auto") -> List[str]:
    """Segmenta um paragrafo em sentencas (usado na leitura do corpus bruto)."""
    text = normalize(text)
    chosen = available_backend(backend)

    if chosen == "spacy":
        try:
            import spacy  # type: ignore

            nlp = _load_spacy()
            if nlp is not None:
                if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                    nlp.add_pipe("sentencizer")
                return [s.text.strip() for s in nlp(text).sents if s.text.strip()]
        except Exception:
            pass
    elif chosen == "nltk" and _load_nltk():
        from nltk.tokenize import sent_tokenize as _nltk_st  # type: ignore

        return [s.strip() for s in _nltk_st(text, language="portuguese") if s.strip()]

    return _regex_sentences(text)


def _regex_sentences(text: str) -> List[str]:
    parts = _SENT_END_RE.split(text)
    out: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # junta quando a quebra caiu logo depois de uma abreviatura conhecida
        if out:
            tail = out[-1].rstrip(".").split()[-1].lower() if out[-1].strip(".") else ""
            tail = _strip_accents(tail)
            if tail in _ABBREV or (len(tail) == 1 and tail.isalpha()):
                out[-1] = out[-1] + " " + part
                continue
        out.append(part)
    return out


def normalize(text: str) -> str:
    """Normalizacao leve: NFC, aspas/travessoes tipograficos e espacos."""
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
        .replace("—", "-").replace("–", "-")
        .replace("\xa0", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def detokenize(tokens: Iterable[str]) -> str:
    """Reconstrucao aproximada — util apenas para inspecao visual."""
    out = ""
    for tok in tokens:
        if out and (tok in ".,;:!?%)]}" or tok.startswith("'")):
            out += tok
        elif out and out[-1] in "([{":
            out += tok
        elif out:
            out += " " + tok
        else:
            out = tok
    return out
