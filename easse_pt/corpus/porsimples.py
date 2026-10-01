"""Leitura e conversao do PorSimples / PorSimpleFul / PorSimplesSent.

O corpus do NILC circula em mais de um formato, dependendo da distribuicao
(pagina do projeto, NILC Download Center, repositorio do PorSimplesSent). Este
modulo detecta e converte os tres formatos mais comuns para o `Dataset` do
EASSE-PT:

  1. TSV/CSV alinhado por sentenca (PorSimplesSent):
     colunas com nomes proximos de original / natural / forte
     (aceita variantes: ori, orig, complex, source / nat / str, strong, forte)

  2. XML do NILC (um arquivo por texto, com <sentence> por versao)

  3. Arquivos-texto paralelos: <nome>.ori / <nome>.nat / <nome>.str,
     uma sentenca por linha, alinhadas pelo numero da linha

Uso tipico:

    easse-pt corpus build data/raw/porsimples --out data/processed/porsimplesent.jsonl

A licenca do PorSimples e do NILC/USP: o corpus NAO e redistribuido junto com
este repositorio. Baixe-o pelos links do plano de pesquisa e coloque em
data/raw/ antes de rodar o comando acima.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

from easse_pt.corpus.dataset import Dataset, Sample
from easse_pt.tokenization import normalize

# Mapeamento tolerante de nomes de coluna -> nivel canonico
_COLUMN_ALIASES: Dict[str, str] = {}
for _canon, _variants in {
    "orig": ["original", "orig", "ori", "complex", "source", "src", "sentenca_original",
             "original_sentence", "sent_original", "s_orig"],
    "natural": ["natural", "nat", "simple_natural", "natural_simplification",
                "sentenca_natural", "s_nat", "nivel_natural"],
    "forte": ["forte", "strong", "str", "simple_strong", "strong_simplification",
              "sentenca_forte", "s_str", "nivel_forte"],
}.items():
    for _v in _variants:
        _COLUMN_ALIASES[_v] = _canon

_XML_SENT_RE = re.compile(r"<sentence[^>]*>(.*?)</sentence>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _canonical(col: str) -> Optional[str]:
    key = re.sub(r"[^a-z_]", "", col.strip().lower().replace(" ", "_").replace("-", "_"))
    return _COLUMN_ALIASES.get(key)


# --------------------------------------------------------------------------- #
# Formatos
# --------------------------------------------------------------------------- #
def read_tabular(path: Path) -> List[Sample]:
    delim = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    samples: List[Sample] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        sniff = fh.readline()
        fh.seek(0)
        if "\t" in sniff and delim == ",":
            delim = "\t"
        reader = csv.DictReader(fh, delimiter=delim)
        if not reader.fieldnames:
            return []
        mapping = {c: _canonical(c) for c in reader.fieldnames}
        if "orig" not in mapping.values():
            raise ValueError(
                f"{path.name}: nenhuma coluna reconhecida como original "
                f"(colunas: {reader.fieldnames})"
            )
        for i, row in enumerate(reader):
            vals = {}
            for col, canon in mapping.items():
                if canon and row.get(col):
                    vals[canon] = normalize(row[col])
            if not vals.get("orig"):
                continue
            samples.append(
                Sample(
                    id=f"{path.stem}-{i:05d}",
                    orig=vals["orig"],
                    natural=vals.get("natural"),
                    forte=vals.get("forte"),
                    source=path.name,
                )
            )
    return samples


def read_xml_triple(path_ori: Path, path_nat: Path, path_str: Path) -> List[Sample]:
    def _sents(p: Path) -> List[str]:
        raw = p.read_text(encoding="utf-8", errors="replace")
        found = _XML_SENT_RE.findall(raw)
        if not found:  # sem tags <sentence>: trata como texto corrido
            from easse_pt.tokenization import sentence_tokenize

            return sentence_tokenize(_TAG_RE.sub(" ", raw))
        return [normalize(_TAG_RE.sub(" ", s)) for s in found]

    ori, nat, stg = _sents(path_ori), _sents(path_nat), _sents(path_str)
    n = min(len(ori), len(nat), len(stg))
    return [
        Sample(id=f"{path_ori.stem}-{i:05d}", orig=ori[i], natural=nat[i], forte=stg[i],
               source=path_ori.name)
        for i in range(n)
        if ori[i].strip()
    ]


def read_parallel_text(stem: Path, suffixes: Dict[str, str]) -> List[Sample]:
    lines: Dict[str, List[str]] = {}
    for level, suf in suffixes.items():
        p = stem.with_suffix(suf)
        if not p.exists():
            continue
        lines[level] = [normalize(x) for x in p.read_text(encoding="utf-8").splitlines()]
    if "orig" not in lines:
        return []
    n = min(len(v) for v in lines.values())
    return [
        Sample(
            id=f"{stem.name}-{i:05d}",
            orig=lines["orig"][i],
            natural=lines.get("natural", [None] * n)[i],
            forte=lines.get("forte", [None] * n)[i],
            source=stem.name,
        )
        for i in range(n)
        if lines["orig"][i].strip()
    ]


# --------------------------------------------------------------------------- #
# Entrada principal
# --------------------------------------------------------------------------- #
def build_dataset(raw_dir: str | Path, name: str = "porsimplesent") -> Dataset:
    """Varre `raw_dir`, detecta o formato e devolve o Dataset consolidado."""
    raw = Path(raw_dir)
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} nao existe. Baixe o PorSimples (links no README) e extraia ali."
        )
    if raw.is_file():
        return Dataset(read_tabular(raw), name=name)

    samples: List[Sample] = []

    for p in sorted(raw.rglob("*")):
        if p.suffix.lower() in (".tsv", ".csv", ".tab") and p.is_file():
            try:
                samples.extend(read_tabular(p))
            except ValueError:
                continue

    # trios XML: <base>_ori.xml / _nat.xml / _str.xml
    for p in sorted(raw.rglob("*ori*.xml")):
        base = p.name.replace("ori", "{}")
        nat, stg = p.with_name(base.format("nat")), p.with_name(base.format("str"))
        if nat.exists() and stg.exists():
            samples.extend(read_xml_triple(p, nat, stg))

    # textos paralelos
    seen = set()
    for p in sorted(raw.rglob("*.ori")):
        if p.stem in seen:
            continue
        seen.add(p.stem)
        samples.extend(
            read_parallel_text(p.with_suffix(""), {"orig": ".ori", "natural": ".nat", "forte": ".str"})
        )

    if not samples:
        raise ValueError(
            f"nenhum formato reconhecido em {raw}. Esperado TSV/CSV alinhado, "
            "trios XML (*ori*.xml/_nat/_str) ou arquivos .ori/.nat/.str"
        )

    # deduplicacao por (orig, natural, forte)
    dedup: Dict[tuple, Sample] = {}
    for s in samples:
        dedup.setdefault((s.orig, s.natural, s.forte), s)
    return Dataset(list(dedup.values()), name=name)


DOWNLOAD_NOTE = """\
O corpus PorSimples e distribuido pelo NILC (ICMC/USP) e nao pode ser
redistribuido neste repositorio. Fontes indicadas no plano de pesquisa:

  * Portal do projeto PorSimples (NILC - ICMC/USP)
  * NILC Download Center — corpus PorSimples completo
  * Repositorio do PorSimplesSent (versao alinhada por sentenca)

Baixe, extraia em data/raw/ e rode:

  easse-pt corpus build data/raw --out data/processed/porsimplesent.jsonl
"""
