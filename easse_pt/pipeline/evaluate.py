"""Avaliacao de uma run: SARI, BLEU, Flesch-Brasil, semantica e atributos.

Produz dois artefatos por run:

  <slug>.scores.json  — metricas agregadas de corpus (linha do benchmark)
  <slug>.per_sent.csv — metricas por sentenca (analise qualitativa de erros)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from easse_pt.corpus.dataset import LEVELS, Dataset
from easse_pt.metrics import (
    bertscore_pt,
    corpus_bleu,
    corpus_features,
    corpus_flesch_brasil,
    corpus_sari,
    flesch_brasil,
    meaning_preservation,
    sentence_features,
    sentence_sari,
)
from easse_pt.pipeline.generate import load_generations
from easse_pt.tokenization import backend_info


def evaluate_run(
    generations_path: str | Path,
    dataset: Dataset,
    ref_levels: Sequence[str] = LEVELS,
    semantic: bool = True,
    per_sentence: bool = True,
    out_dir: Optional[str | Path] = None,
) -> Dict:
    """Avalia um arquivo de geracoes contra as referencias do dataset."""
    gen_path = Path(generations_path)
    recs = load_generations(gen_path)
    by_id = {str(r["id"]): r for r in recs}

    samples = [s for s in dataset if str(s.id) in by_id]
    if not samples:
        raise ValueError(f"nenhuma sentenca de {gen_path.name} bate com o dataset {dataset.name}")

    orig_sents = [s.orig for s in samples]
    sys_sents = [by_id[str(s.id)].get("sys") or "" for s in samples]

    levels = [lv for lv in ref_levels if all(getattr(s, lv) for s in samples)]
    if not levels:
        raise ValueError("nenhum nivel de referencia completo para estas sentencas")
    refs_sents: List[List[str]] = [[getattr(s, lv) for s in samples] for lv in levels]

    scores: Dict[str, object] = {
        "run": gen_path.stem,
        "dataset": dataset.name,
        "n": len(samples),
        "ref_levels": list(levels),
        "model": recs[0].get("model"),
        "strategy": recs[0].get("strategy"),
        "level": recs[0].get("level"),
        "n_errors": sum(1 for r in recs if r.get("error")),
    }
    scores.update(backend_info())

    # ---- metricas principais -------------------------------------------- #
    scores.update(corpus_sari(orig_sents, sys_sents, refs_sents))
    scores.update(corpus_bleu(sys_sents, refs_sents))

    scores["flesch_sys"] = corpus_flesch_brasil(sys_sents)
    scores["flesch_orig"] = corpus_flesch_brasil(orig_sents)
    for lv, col in zip(levels, refs_sents):
        scores[f"flesch_ref_{lv}"] = corpus_flesch_brasil(col)
    scores["flesch_gain"] = scores["flesch_sys"] - scores["flesch_orig"]

    scores.update(corpus_features(orig_sents, sys_sents))

    # ---- preservacao semantica (opcional) -------------------------------- #
    # Falha aqui (download de modelo, incompatibilidade de versao) nao pode
    # derrubar SARI/BLEU/Flesch ja calculados: registra o erro e segue.
    if semantic:
        bs = mp = None
        try:
            bs = bertscore_pt(sys_sents, refs_sents)
            if bs:
                scores.update(bs)
        except Exception as exc:  # noqa: BLE001
            scores["bertscore_error"] = f"{type(exc).__name__}: {exc}"[:500]
        try:
            mp = meaning_preservation(orig_sents, sys_sents)
            if mp:
                scores.update(mp)
        except Exception as exc:  # noqa: BLE001
            scores["meaning_error"] = f"{type(exc).__name__}: {exc}"[:500]
        if not bs and not mp:
            scores["semantic_available"] = False

    # ---- latencia -------------------------------------------------------- #
    lats = [r.get("latency_s") for r in recs if isinstance(r.get("latency_s"), (int, float))]
    if lats:
        scores["latency_mean_s"] = sum(lats) / len(lats)
        scores["latency_total_s"] = sum(lats)

    out_dir = Path(out_dir) if out_dir else gen_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{gen_path.stem}.scores.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if per_sentence:
        _write_per_sentence(out_dir / f"{gen_path.stem}.per_sent.csv",
                            samples, sys_sents, levels, by_id)

    return scores


def _write_per_sentence(path: Path, samples, sys_sents, levels, by_id) -> None:
    import csv

    fields = [
        "id", "orig", "sys", *[f"ref_{lv}" for lv in levels],
        "sari", "sari_keep", "sari_del", "sari_add",
        "flesch_orig", "flesch_sys", "flesch_gain",
        "compression_ratio", "sentence_splits", "levenshtein_sim", "exact_copy",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for s, sys_sent in zip(samples, sys_sents):
            refs = [getattr(s, lv) for lv in levels]
            row = {"id": s.id, "orig": s.orig, "sys": sys_sent,
                   "error": by_id[str(s.id)].get("error", "")}
            row.update({f"ref_{lv}": getattr(s, lv) for lv in levels})
            row.update({k: round(v, 3) for k, v in sentence_sari(s.orig, sys_sent, refs).items()})
            fo, fs = flesch_brasil(s.orig), flesch_brasil(sys_sent)
            row.update({"flesch_orig": round(fo, 2), "flesch_sys": round(fs, 2),
                        "flesch_gain": round(fs - fo, 2)})
            feats = sentence_features(s.orig, sys_sent)
            row.update({k: round(feats[k], 3) for k in
                        ("compression_ratio", "sentence_splits", "levenshtein_sim", "exact_copy")})
            w.writerow(row)


def evaluate_references(dataset: Dataset, semantic: bool = False) -> Dict[str, Dict]:
    """Teto humano: avalia cada nivel de referencia usando o outro como alvo.

    Da a escala em que ler as notas das LLMs — um SARI de 40 so significa algo
    ao lado do SARI que a propria referencia humana obtem.
    """
    out: Dict[str, Dict] = {}
    levels = dataset.available_levels()
    for lv in levels:
        others = [o for o in levels if o != lv]
        if not others:
            continue
        sys_sents = [getattr(s, lv) for s in dataset]
        refs = [[getattr(s, o) for s in dataset] for o in others]
        row = {
            "run": f"REFERENCIA_{lv}",
            "model": "[referencia humana]",
            "strategy": "—",
            "level": lv,
            "n": len(dataset),
            "ref_levels": others,
        }
        row.update(corpus_sari(dataset.orig_sents, sys_sents, refs))
        row.update(corpus_bleu(sys_sents, refs))
        row["flesch_sys"] = corpus_flesch_brasil(sys_sents)
        row["flesch_orig"] = corpus_flesch_brasil(dataset.orig_sents)
        row["flesch_gain"] = row["flesch_sys"] - row["flesch_orig"]
        row.update(corpus_features(dataset.orig_sents, sys_sents))
        if semantic:
            mp = meaning_preservation(dataset.orig_sents, sys_sents)
            if mp:
                row.update(mp)
        out[lv] = row
    return out
