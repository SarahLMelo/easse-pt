"""Consolidacao do benchmark: tabela comparativa em CSV e Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

COLUMNS = [
    ("model", "Modelo"),
    ("strategy", "Prompt"),
    ("level", "Nivel"),
    ("sari", "SARI"),
    ("sari_keep", "keep"),
    ("sari_del", "del"),
    ("sari_add", "add"),
    ("bleu", "BLEU"),
    ("flesch_sys", "Flesch-BR"),
    ("flesch_gain", "ΔFlesch"),
    ("bertscore_f1", "BERTScore-PT"),
    ("meaning_sim_mean", "Sem.(orig)"),
    ("compression_ratio", "Compr."),
    ("exact_copy", "Copia %"),
    ("latency_mean_s", "s/sent"),
]

_FLOAT_COLS = {c for c, _ in COLUMNS} - {"model", "strategy", "level"}


def collect_scores(out_dir: str | Path = "data/outputs") -> List[Dict]:
    rows = []
    for p in sorted(Path(out_dir).glob("*.scores.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows


def to_dataframe(rows: Sequence[Dict]):
    import pandas as pd

    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    present = [c for c, _ in COLUMNS if c in df.columns]
    extra = [c for c in ("run", "n", "n_errors") if c in df.columns]
    df = df[present + extra]
    if "sari" in df.columns:
        df = df.sort_values("sari", ascending=False)
    return df


def to_markdown(rows: Sequence[Dict], title: str = "Benchmark EASSE-PT") -> str:
    df = to_dataframe(rows)
    if df.empty:
        return f"# {title}\n\n_Nenhum resultado encontrado._\n"

    header = [label for col, label in COLUMNS if col in df.columns]
    keys = [col for col, _ in COLUMNS if col in df.columns]

    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for _, r in df.iterrows():
        cells = []
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and v != v):
                cells.append("—")
            elif k in _FLOAT_COLS and isinstance(v, (int, float)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "**Leitura das colunas.** SARI e a metrica principal (maior = melhor); "
        "`keep`/`del`/`add` sao as parcelas por operacao de edicao. BLEU mede "
        "proximidade das referencias e penaliza reescritas ousadas — serve de "
        "contrapeso, nao de criterio. Flesch-BR maior = texto mais facil, e "
        "ΔFlesch e o ganho de legibilidade sobre a sentenca original. "
        "BERTScore-PT e Sem.(orig) medem preservacao de significado (contra as "
        "referencias e contra a sentenca fonte, respectivamente). `Copia %` alta "
        "com ΔFlesch baixo indica modelo que apenas repete a entrada.",
    ]
    return "\n".join(lines) + "\n"


def write_report(
    out_dir: str | Path = "data/outputs",
    report_dir: str | Path = "data/outputs",
    title: str = "Benchmark EASSE-PT",
    reference_rows: Optional[Sequence[Dict]] = None,
) -> Dict[str, Path]:
    rows = list(collect_scores(out_dir))
    if reference_rows:
        rows = list(reference_rows) + rows

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    df = to_dataframe(rows)
    csv_path = report_dir / "benchmark.csv"
    md_path = report_dir / "benchmark.md"
    if not df.empty:
        df.to_csv(csv_path, index=False, encoding="utf-8")
    md_path.write_text(to_markdown(rows, title=title), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}
