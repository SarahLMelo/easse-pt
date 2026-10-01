"""Estrutura de dados do benchmark, no layout esperado pelo EASSE.

O EASSE trabalha com tres listas paralelas:

    orig_sents  : [str]            sentencas originais (entrada do sistema)
    sys_sents   : [str]            saidas do sistema
    refs_sents  : [[str], [str]]   um vetor por CONJUNTO de referencia

No EASSE-PT os dois conjuntos sao os dois niveis do PorSimples:
`natural` (leitores em geral) e `forte` (leitores com baixo nivel de
alfabetizacao). Em disco, o dataset vive como um JSONL — um registro por
sentenca, facil de versionar e de inspecionar — e pode ser exportado para os
arquivos texto paralelos que o EASSE original consome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

LEVELS = ("natural", "forte")


@dataclass
class Sample:
    id: str
    orig: str
    natural: Optional[str] = None
    forte: Optional[str] = None
    source: str = ""
    meta: Dict = field(default_factory=dict)

    def refs(self, levels: Sequence[str] = LEVELS) -> List[str]:
        return [getattr(self, lv) for lv in levels if getattr(self, lv)]


@dataclass
class Dataset:
    samples: List[Sample]
    name: str = "porsimplesent"

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    @property
    def ids(self) -> List[str]:
        return [s.id for s in self.samples]

    @property
    def orig_sents(self) -> List[str]:
        return [s.orig for s in self.samples]

    def refs_sents(self, levels: Sequence[str] = LEVELS) -> List[List[str]]:
        """Matriz multi-referencia para o SARI: um vetor por nivel."""
        out = []
        for lv in levels:
            col = [getattr(s, lv) for s in self.samples]
            if any(c is None for c in col):
                raise ValueError(
                    f"nivel '{lv}' ausente em {sum(c is None for c in col)} sentencas; "
                    "use --levels para restringir aos niveis completos"
                )
            out.append(col)
        return out

    def available_levels(self) -> List[str]:
        return [lv for lv in LEVELS if all(getattr(s, lv) for s in self.samples)]

    def subset(self, n: Optional[int] = None, seed: int = 13) -> "Dataset":
        """Amostra aleatoria reprodutivel — util para depurar prompts sem gastar tempo."""
        if n is None or n >= len(self.samples):
            return self
        import random

        rng = random.Random(seed)
        picked = sorted(rng.sample(range(len(self.samples)), n))
        return Dataset([self.samples[i] for i in picked], name=f"{self.name}-n{n}")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_jsonl(cls, path: str | Path, name: Optional[str] = None) -> "Dataset":
        path = Path(path)
        samples = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                samples.append(
                    Sample(
                        id=str(d.get("id", len(samples))),
                        orig=d["orig"],
                        natural=d.get("natural"),
                        forte=d.get("forte"),
                        source=d.get("source", ""),
                        meta=d.get("meta", {}),
                    )
                )
        return cls(samples, name=name or path.stem)

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for s in self.samples:
                fh.write(
                    json.dumps(
                        {
                            "id": s.id, "orig": s.orig, "natural": s.natural,
                            "forte": s.forte, "source": s.source, "meta": s.meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return path

    def to_easse_files(self, out_dir: str | Path) -> Dict[str, Path]:
        """Exporta os arquivos paralelos consumidos pelo EASSE original.

        Gera <name>.orig, <name>.natural e <name>.forte — uma sentenca por linha,
        alinhadas pelo numero da linha.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}

        def _write(suffix: str, lines: Sequence[str]) -> None:
            p = out_dir / f"{self.name}.{suffix}"
            p.write_text("\n".join(line.replace("\n", " ") for line in lines) + "\n",
                         encoding="utf-8")
            written[suffix] = p

        _write("orig", self.orig_sents)
        for lv in self.available_levels():
            _write(lv, [getattr(s, lv) for s in self.samples])
        return written

    def stats(self) -> Dict[str, float]:
        from easse_pt.metrics.readability import corpus_flesch_brasil

        out: Dict[str, float] = {"n_samples": float(len(self.samples))}
        out["flesch_orig"] = corpus_flesch_brasil(self.orig_sents)
        for lv in self.available_levels():
            out[f"flesch_{lv}"] = corpus_flesch_brasil([getattr(s, lv) for s in self.samples])
        return out
