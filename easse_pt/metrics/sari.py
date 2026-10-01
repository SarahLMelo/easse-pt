"""SARI (Xu et al., 2016) com tokenizacao em Portugues.

Implementacao fiel a de referencia do EASSE / star (Xu et al.), com a unica
mudanca prevista no plano: os n-gramas sao contados sobre a tokenizacao PT-BR
de `easse_pt.tokenization`, nao sobre o tokenizador ingles.

SARI compara a saida do sistema (C) com a entrada original (S) e com N
referencias humanas (R), pontuando tres operacoes de edicao:

    KEEP : n-gramas mantidos de S que as referencias tambem mantem  (F1)
    DEL  : n-gramas apagados de S que as referencias tambem apagam  (precisao)
    ADD  : n-gramas novos em C que aparecem nas referencias         (F1)

A nota final e a media de KEEP/DEL/ADD, promediada sobre n = 1..4.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from easse_pt.tokenization import tokenize

NGRAM_ORDER = 4


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _sari_ngram(
    sgrams: List[tuple], cgrams: List[tuple], rgramslist: List[List[tuple]], numref: int
):
    rgramsall = [g for rgrams in rgramslist for g in rgrams]
    rgramcounter = Counter(rgramsall)

    sgramcounter = Counter(sgrams)
    sgramcounter_rep = Counter({k: v * numref for k, v in sgramcounter.items()})
    cgramcounter = Counter(cgrams)
    cgramcounter_rep = Counter({k: v * numref for k, v in cgramcounter.items()})

    # ---------------- KEEP ----------------
    keepgramcounter_rep = sgramcounter_rep & cgramcounter_rep
    keepgramcountergood_rep = keepgramcounter_rep & rgramcounter
    keepgramcounterall_rep = sgramcounter_rep & rgramcounter

    keeptmp1 = 0.0
    keeptmp2 = 0.0
    for g in keepgramcountergood_rep:
        keeptmp1 += keepgramcountergood_rep[g] / keepgramcounter_rep[g]
        keeptmp2 += keepgramcountergood_rep[g]

    keep_p = 1.0
    keep_r = 1.0
    if len(keepgramcounter_rep) > 0:
        keep_p = keeptmp1 / len(keepgramcounter_rep)
    if len(keepgramcounterall_rep) > 0:
        keep_r = keeptmp2 / sum(keepgramcounterall_rep.values())
    keep_f = 0.0
    if keep_p > 0 or keep_r > 0:
        keep_f = 2 * keep_p * keep_r / (keep_p + keep_r)

    # ---------------- DELETE ----------------
    delgramcounter_rep = sgramcounter_rep - cgramcounter_rep
    delgramcountergood_rep = delgramcounter_rep - rgramcounter
    delgramcounterall_rep = sgramcounter_rep - rgramcounter

    deltmp1 = 0.0
    for g in delgramcountergood_rep:
        deltmp1 += delgramcountergood_rep[g] / delgramcounter_rep[g]
    del_p = 1.0
    if len(delgramcounter_rep) > 0:
        del_p = deltmp1 / len(delgramcounter_rep)
    del_score = del_p  # SARI usa apenas a precisao da delecao

    # ---------------- ADD ----------------
    addgramcounter = set(cgramcounter) - set(sgramcounter)
    addgramcountergood = addgramcounter & set(rgramcounter)
    addgramcounterall = set(rgramcounter) - set(sgramcounter)

    add_p = 1.0
    add_r = 1.0
    if len(addgramcounter) > 0:
        add_p = len(addgramcountergood) / len(addgramcounter)
    if len(addgramcounterall) > 0:
        add_r = len(addgramcountergood) / len(addgramcounterall)
    add_f = 0.0
    if add_p > 0 or add_r > 0:
        add_f = 2 * add_p * add_r / (add_p + add_r)

    return keep_f, del_score, add_f


def sentence_sari(
    orig_sent: str,
    sys_sent: str,
    ref_sents: Sequence[str],
    lowercase: bool = True,
    tokenizer_backend: str = "auto",
) -> Dict[str, float]:
    """SARI de uma sentenca. Retorna a nota final e as parcelas keep/del/add (0-100)."""
    if not ref_sents:
        raise ValueError("SARI exige ao menos uma referencia")

    s_tok = tokenize(orig_sent, lowercase=lowercase, backend=tokenizer_backend)
    c_tok = tokenize(sys_sent, lowercase=lowercase, backend=tokenizer_backend)
    r_toks = [tokenize(r, lowercase=lowercase, backend=tokenizer_backend) for r in ref_sents]
    numref = len(r_toks)

    keep, dele, add = [], [], []
    for n in range(1, NGRAM_ORDER + 1):
        k, d, a = _sari_ngram(
            _ngrams(s_tok, n), _ngrams(c_tok, n), [_ngrams(r, n) for r in r_toks], numref
        )
        keep.append(k)
        dele.append(d)
        add.append(a)

    keep_avg = sum(keep) / NGRAM_ORDER
    del_avg = sum(dele) / NGRAM_ORDER
    add_avg = sum(add) / NGRAM_ORDER
    return {
        "sari": 100 * (keep_avg + del_avg + add_avg) / 3,
        "sari_keep": 100 * keep_avg,
        "sari_del": 100 * del_avg,
        "sari_add": 100 * add_avg,
    }


def corpus_sari(
    orig_sents: Sequence[str],
    sys_sents: Sequence[str],
    refs_sents: Sequence[Sequence[str]],
    lowercase: bool = True,
    tokenizer_backend: str = "auto",
) -> Dict[str, float]:
    """SARI de corpus = media das notas por sentenca (padrao do EASSE).

    `refs_sents` segue o layout do EASSE: lista de conjuntos de referencia, em que
    `refs_sents[j][i]` e a j-esima referencia da i-esima sentenca. No EASSE-PT os
    dois conjuntos sao a referencia Natural e a referencia Forte do PorSimpleFul.
    """
    n = len(orig_sents)
    if len(sys_sents) != n:
        raise ValueError(f"orig ({n}) e sys ({len(sys_sents)}) tem tamanhos diferentes")
    for j, refs in enumerate(refs_sents):
        if len(refs) != n:
            raise ValueError(f"conjunto de referencia {j} tem {len(refs)} sentencas, esperado {n}")

    acc = {"sari": 0.0, "sari_keep": 0.0, "sari_del": 0.0, "sari_add": 0.0}
    for i in range(n):
        refs_i = [refs[i] for refs in refs_sents]
        scores = sentence_sari(
            orig_sents[i], sys_sents[i], refs_i,
            lowercase=lowercase, tokenizer_backend=tokenizer_backend,
        )
        for k in acc:
            acc[k] += scores[k]
    return {k: v / n for k, v in acc.items()} if n else acc
