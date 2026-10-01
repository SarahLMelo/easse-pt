"""Atributos de controle / quality estimation, no espirito do EASSE.

Sao descritores do QUE o sistema fez, nao notas de qualidade: taxa de
compressao, divisao de sentencas, sobreposicao lexical, copia literal. Servem
para a analise qualitativa de erros do Mes 3 — por exemplo, distinguir um
modelo que simplifica de um que apenas copia a entrada.
"""

from __future__ import annotations

from typing import Dict, Sequence

from easse_pt.tokenization import sentence_tokenize, tokenize


def _lev_ratio(a: Sequence[str], b: Sequence[str]) -> float:
    """Similaridade de Levenshtein normalizada (0-1) em nivel de token."""
    if not a and not b:
        return 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def sentence_features(orig: str, sys: str, tokenizer_backend: str = "auto") -> Dict[str, float]:
    o = tokenize(orig, backend=tokenizer_backend)
    s = tokenize(sys, backend=tokenizer_backend)
    o_set, s_set = set(o), set(s)
    return {
        "compression_ratio": len(s) / len(o) if o else 0.0,
        "sentence_splits": len(sentence_tokenize(sys, backend=tokenizer_backend))
        / max(1, len(sentence_tokenize(orig, backend=tokenizer_backend))),
        "lexical_copy": len(o_set & s_set) / len(s_set) if s_set else 0.0,
        "added_words": len(s_set - o_set) / len(s_set) if s_set else 0.0,
        "deleted_words": len(o_set - s_set) / len(o_set) if o_set else 0.0,
        "levenshtein_sim": _lev_ratio(o, s),
        "exact_copy": float(" ".join(o) == " ".join(s)),
        "is_empty": float(len(s) == 0),
    }


def corpus_features(
    orig_sents: Sequence[str], sys_sents: Sequence[str], tokenizer_backend: str = "auto"
) -> Dict[str, float]:
    rows = [sentence_features(o, s, tokenizer_backend) for o, s in zip(orig_sents, sys_sents)]
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
    out["exact_copy"] *= 100  # em %
    out["is_empty"] *= 100
    return out
