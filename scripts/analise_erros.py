#!/usr/bin/env python3
"""Triagem quantitativa de erros a partir dos CSVs por sentenca.

Classifica cada saida em categorias de falha e imprime a distribuicao por run,
alem de listar exemplos de cada categoria. E o ponto de partida da analise
qualitativa do Mes 3 — a leitura dos exemplos continua sendo manual.

Categorias:
  copia            saida identica (ou quase) a entrada — o modelo nao simplificou
  sem_ganho        reescreveu, mas o Flesch-Brasil nao melhorou
  truncada         saida muito mais curta que a referencia mais curta
  inflada          saida mais longa que a original (simplificacao que complica)
  vazia/erro       falha de geracao registrada pelo pipeline
  ok               nenhuma das anteriores

Uso:
    python scripts/analise_erros.py data/outputs/*.per_sent.csv
    python scripts/analise_erros.py --exemplos 3 data/outputs/*.per_sent.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIAS = ["vazia/erro", "copia", "sem_ganho", "truncada", "inflada", "ok"]


def classificar(row: dict) -> str:
    if row.get("error") or not (row.get("sys") or "").strip():
        return "vazia/erro"
    try:
        lev = float(row["levenshtein_sim"])
        ganho = float(row["flesch_gain"])
        compr = float(row["compression_ratio"])
    except (KeyError, ValueError):
        return "ok"
    if float(row.get("exact_copy", 0)) == 1.0 or lev > 0.95:
        return "copia"
    if ganho <= 0:
        return "sem_ganho"
    if compr < 0.35:
        return "truncada"
    if compr > 1.15:
        return "inflada"
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+", help="arquivos *.per_sent.csv (aceita glob)")
    ap.add_argument("--exemplos", type=int, default=2, help="exemplos por categoria (0 desliga)")
    args = ap.parse_args(argv)

    paths = []
    for pattern in args.csvs:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("nenhum CSV encontrado", file=sys.stderr)
        return 1

    for path in paths:
        rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
        if not rows:
            continue
        contagem = Counter()
        exemplos = defaultdict(list)
        for r in rows:
            cat = classificar(r)
            contagem[cat] += 1
            if len(exemplos[cat]) < args.exemplos:
                exemplos[cat].append(r)

        n = len(rows)
        print(f"\n{'=' * 78}\n{Path(path).stem}  ({n} sentencas)\n{'=' * 78}")
        for cat in CATEGORIAS:
            c = contagem.get(cat, 0)
            if c:
                barra = "#" * int(40 * c / n)
                print(f"  {cat:12} {c:4d}  {100 * c / n:5.1f}%  {barra}")

        for cat in CATEGORIAS:
            if cat in ("ok",) or not exemplos.get(cat):
                continue
            print(f"\n  --- exemplos: {cat} ---")
            for r in exemplos[cat]:
                print(f"    [{r['id']}] SARI {r.get('sari', '?')} "
                      f"ΔFlesch {r.get('flesch_gain', '?')}")
                print(f"      orig: {r['orig'][:150]}")
                print(f"      sys : {(r['sys'] or '(vazio)')[:150]}")
                if r.get("error"):
                    print(f"      erro: {r['error']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
