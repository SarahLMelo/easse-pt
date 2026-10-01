"""Preservacao semantica com embeddings em Portugues (BERTScore-PT).

Dois backends, ambos opcionais (dependencia `semantic` no pyproject):

  bertscore  -> pacote `bert_score` com BERTimbau (neuralmind/bert-base-portuguese-cased)
  sbert      -> cosseno entre embeddings de sentenca (paraphrase-multilingual-MiniLM-L12-v2)

Se nenhuma das bibliotecas estiver instalada, as funcoes retornam None em vez
de quebrar o pipeline — as demais metricas continuam sendo calculadas e o
relatorio marca a coluna como indisponivel.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_BERTSCORE_MODEL = "neuralmind/bert-base-portuguese-cased"
DEFAULT_SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def is_available(backend: str = "auto") -> bool:
    if backend in ("auto", "bertscore"):
        try:
            import bert_score  # noqa: F401

            return True
        except Exception:
            if backend == "bertscore":
                return False
    if backend in ("auto", "sbert"):
        try:
            import sentence_transformers  # noqa: F401

            return True
        except Exception:
            return False
    return False


@lru_cache(maxsize=2)
def _sbert(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def _bert_scorer(model_name: str, batch_size: int):
    from bert_score import BERTScorer  # type: ignore

    scorer = BERTScorer(
        model_type=model_name, num_layers=10, lang="pt",
        batch_size=batch_size, rescale_with_baseline=False,
    )
    # O BERTimbau nao define model_max_length no tokenizador (fica ~1e30) e o
    # bert_score repassa esse valor ao truncar -> "OverflowError: int too big
    # to convert". Modelos BERT aceitam no maximo 512 tokens.
    tok = getattr(scorer, "_tokenizer", None)
    if tok is not None and (getattr(tok, "model_max_length", 0) or 0) > 100_000:
        tok.model_max_length = 512
    return scorer


def bertscore_pt(
    sys_sents: Sequence[str],
    refs_sents: Sequence[Sequence[str]],
    model_name: str = DEFAULT_BERTSCORE_MODEL,
    batch_size: int = 32,
) -> Optional[Dict[str, float]]:
    """BERTScore (P/R/F1) com modelo portugues, multi-referencia (melhor referencia)."""
    try:
        import bert_score  # noqa: F401  # type: ignore
    except Exception:
        log.warning("bert_score indisponivel — pule com `pip install -r requirements-semantic.txt`")
        return None

    cands: List[str] = list(sys_sents)
    refs: List[List[str]] = [[ref_set[i] for ref_set in refs_sents] for i in range(len(cands))]

    scorer = _bert_scorer(model_name, batch_size)
    P, R, F = scorer.score(cands, refs, verbose=False, batch_size=batch_size)
    return {
        "bertscore_p": float(P.mean()) * 100,
        "bertscore_r": float(R.mean()) * 100,
        "bertscore_f1": float(F.mean()) * 100,
    }


def sbert_similarity(
    sys_sents: Sequence[str],
    other_sents: Sequence[str],
    model_name: str = DEFAULT_SBERT_MODEL,
    batch_size: int = 32,
) -> Optional[List[float]]:
    """Cosseno par a par entre `sys_sents` e `other_sents` (0-1)."""
    try:
        from sentence_transformers import util  # type: ignore
    except Exception:
        log.warning("sentence-transformers indisponivel — similaridade semantica omitida")
        return None

    model = _sbert(model_name)
    a = model.encode(list(sys_sents), batch_size=batch_size, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)
    b = model.encode(list(other_sents), batch_size=batch_size, convert_to_tensor=True,
                     normalize_embeddings=True, show_progress_bar=False)
    return [float(util.cos_sim(a[i], b[i])) for i in range(len(sys_sents))]


def meaning_preservation(
    orig_sents: Sequence[str],
    sys_sents: Sequence[str],
    model_name: str = DEFAULT_SBERT_MODEL,
) -> Optional[Dict[str, float]]:
    """Quanto do sentido do ORIGINAL a simplificacao preserva.

    Complementa o BERTScore contra as referencias: um modelo pode acertar o
    estilo da referencia e ainda assim perder fatos da sentenca fonte — a perda
    de significado apontada por Zhang et al. (2026) como principal risco da
    simplificacao graduada.
    """
    sims = sbert_similarity(sys_sents, orig_sents, model_name=model_name)
    if sims is None:
        return None
    return {
        "meaning_sim_mean": 100 * sum(sims) / len(sims),
        "meaning_sim_min": 100 * min(sims),
        "meaning_drop_rate": 100 * sum(1 for s in sims if s < 0.80) / len(sims),
    }
