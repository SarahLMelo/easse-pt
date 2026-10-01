"""BLEU multi-referencia sobre a tokenizacao PT-BR.

Quando o sacrebleu esta instalado ele e usado como motor de contagem — mas
apenas como motor: o texto ja chega tokenizado pelo `easse_pt.tokenization` e
passamos `tokenize="none"` para que o tokenizador ingles padrao (13a) nao seja
aplicado por cima. Assim BLEU e SARI enxergam exatamente os mesmos n-gramas.

Sem o sacrebleu, ha uma implementacao interna de BLEU-4 (Papineni et al., 2002)
com penalidade de brevidade e selecao da referencia de comprimento mais
proximo, equivalente ao sacrebleu com `tokenize="none"` e suavizacao `exp`. O
campo `bleu_engine` no resultado registra qual caminho foi usado — informacao
que deve constar do relatorio do TCC.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from easse_pt.tokenization import tokenize

NGRAM_ORDER = 4


def _has_sacrebleu() -> bool:
    try:
        import sacrebleu  # noqa: F401

        return True
    except Exception:
        return False


def _pretok(text: str, lowercase: bool, backend: str) -> str:
    return " ".join(tokenize(text, lowercase=lowercase, backend=backend))


# --------------------------------------------------------------------------- #
# Implementacao interna (usada quando nao ha sacrebleu)
# --------------------------------------------------------------------------- #
def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _closest_ref_len(hyp_len: int, ref_lens: Sequence[int]) -> int:
    return min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))


def _bleu_fallback(hyps: Sequence[List[str]], refs: Sequence[Sequence[List[str]]]) -> Dict[str, float]:
    correct = [0] * NGRAM_ORDER
    total = [0] * NGRAM_ORDER
    hyp_len = 0
    ref_len = 0

    for hyp, ref_set in zip(hyps, refs):
        hyp_len += len(hyp)
        ref_len += _closest_ref_len(len(hyp), [len(r) for r in ref_set])
        for n in range(1, NGRAM_ORDER + 1):
            hyp_counts = _ngram_counts(hyp, n)
            max_ref: Counter = Counter()
            for r in ref_set:
                rc = _ngram_counts(r, n)
                for g, c in rc.items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            total[n - 1] += max(0, len(hyp) - n + 1)
            correct[n - 1] += sum(min(c, max_ref[g]) for g, c in hyp_counts.items())

    # precisoes com suavizacao exponencial (mesma do sacrebleu para ordens zeradas)
    precisions = []
    smooth = 1.0
    for n in range(NGRAM_ORDER):
        if total[n] == 0:
            precisions.append(0.0)
        elif correct[n] == 0:
            smooth *= 2
            precisions.append(100.0 / (smooth * total[n]))
        else:
            precisions.append(100.0 * correct[n] / total[n])

    if min(precisions) <= 0:
        score = 0.0
        bp = 0.0
    else:
        bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len) if hyp_len else 0.0
        log_avg = sum(math.log(p) for p in precisions) / NGRAM_ORDER
        score = bp * math.exp(log_avg)

    return {
        "bleu": score,
        "bleu_bp": bp,
        "bleu_ratio": hyp_len / ref_len if ref_len else 0.0,
        "bleu_engine": "interno",
    }


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def corpus_bleu(
    sys_sents: Sequence[str],
    refs_sents: Sequence[Sequence[str]],
    lowercase: bool = True,
    tokenizer_backend: str = "auto",
) -> Dict[str, float]:
    """BLEU de corpus. `refs_sents[j][i]` = j-esima referencia da i-esima sentenca."""
    if _has_sacrebleu():
        import sacrebleu

        hyps = [_pretok(s, lowercase, tokenizer_backend) for s in sys_sents]
        refs = [[_pretok(r, lowercase, tokenizer_backend) for r in rs] for rs in refs_sents]
        bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize="none", force=True)
        return {
            "bleu": bleu.score,
            "bleu_bp": bleu.bp,
            "bleu_ratio": bleu.sys_len / bleu.ref_len if bleu.ref_len else 0.0,
            "bleu_engine": "sacrebleu",
        }

    hyps_t = [tokenize(s, lowercase=lowercase, backend=tokenizer_backend) for s in sys_sents]
    refs_t = [
        [tokenize(rs[i], lowercase=lowercase, backend=tokenizer_backend) for rs in refs_sents]
        for i in range(len(sys_sents))
    ]
    return _bleu_fallback(hyps_t, refs_t)


def sentence_bleu(
    sys_sent: str,
    ref_sents: Sequence[str],
    lowercase: bool = True,
    tokenizer_backend: str = "auto",
) -> float:
    """BLEU de uma sentenca (suavizado — para analise de erros)."""
    if _has_sacrebleu():
        import sacrebleu

        hyp = _pretok(sys_sent, lowercase, tokenizer_backend)
        refs = [_pretok(r, lowercase, tokenizer_backend) for r in ref_sents]
        return sacrebleu.sentence_bleu(hyp, refs, tokenize="none").score

    hyp_t = tokenize(sys_sent, lowercase=lowercase, backend=tokenizer_backend)
    refs_t = [tokenize(r, lowercase=lowercase, backend=tokenizer_backend) for r in ref_sents]
    return _bleu_fallback([hyp_t], [refs_t])["bleu"]
