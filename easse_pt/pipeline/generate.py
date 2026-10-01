"""Geracao das simplificacoes pelas LLMs.

Uma "run" = (modelo x estrategia de prompting x nivel de simplificacao) sobre
um dataset. O resultado e um JSONL em data/outputs/, com uma linha por
sentenca contendo a saida crua do modelo, a saida pos-processada e metadados
(latencia, tokens). O arquivo cru e preservado de proposito: a analise
qualitativa de erros do Mes 3 depende de ver o raciocinio do CoT.

A execucao e retomavel: ao reprocessar uma run, as sentencas ja geradas sao
lidas do arquivo existente e apenas as faltantes vao ao modelo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from easse_pt.corpus.dataset import Dataset
from easse_pt.llm.base import LLMProvider
from easse_pt.prompts import build_prompt, postprocess
from easse_pt.tokenization import backend_info


@dataclass
class RunSpec:
    model: str          # spec completa, ex. 'ollama/llama3.1:8b'
    strategy: str
    level: str
    dataset: str

    @property
    def slug(self) -> str:
        m = self.model.replace("/", "_").replace(":", "-")
        return f"{self.dataset}__{m}__{self.strategy}__{self.level}"


def output_path(spec: RunSpec, out_dir: str | Path = "data/outputs") -> Path:
    return Path(out_dir) / f"{spec.slug}.jsonl"


def _load_existing(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    done: Dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("sys") or rec.get("error"):
                done[str(rec.get("id"))] = rec
    return done


def run_generation(
    dataset: Dataset,
    provider: LLMProvider,
    strategy: str = "zero_shot",
    level: str = "forte",
    out_dir: str | Path = "data/outputs",
    resume: bool = True,
    limit: Optional[int] = None,
    progress: bool = True,
) -> Path:
    """Executa uma run e devolve o caminho do JSONL gerado."""
    ds = dataset.subset(limit) if limit else dataset
    spec = RunSpec(
        model=f"{provider.name}/{provider.model}", strategy=strategy,
        level=level, dataset=ds.name,
    )
    path = output_path(spec, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(path) if resume else {}
    pending = [s for s in ds if str(s.id) not in existing]

    iterator = pending
    if progress and pending:
        try:
            from tqdm import tqdm

            iterator = tqdm(pending, desc=spec.slug, unit="sent")
        except ImportError:
            pass

    records: List[dict] = [existing[str(s.id)] for s in ds if str(s.id) in existing]

    for sample in iterator:
        prompt = build_prompt(sample.orig, strategy=strategy, level=level)
        rec: Dict = {
            "id": sample.id,
            "orig": sample.orig,
            "strategy": strategy,
            "level": level,
            "model": spec.model,
        }
        try:
            resp = provider.generate(prompt.text, system=prompt.system)
            rec.update(
                {
                    "raw": resp.text,
                    "sys": postprocess(resp.text, strategy=strategy),
                    "latency_s": round(resp.latency_s, 3),
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                }
            )
            if not rec["sys"]:
                rec["error"] = "saida vazia apos pos-processamento"
                rec["sys"] = sample.orig  # fallback de copia: nao quebra as metricas
        except Exception as exc:  # rede, timeout, modelo ausente
            rec.update({"raw": "", "sys": sample.orig, "error": f"{type(exc).__name__}: {exc}"})
        records.append(rec)

    # ordena na ordem do dataset e reescreve o arquivo inteiro
    order = {str(s.id): i for i, s in enumerate(ds)}
    records.sort(key=lambda r: order.get(str(r["id"]), 10**9))
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "spec": asdict(spec),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n": len(records),
        "n_errors": sum(1 for r in records if r.get("error")),
        "generation_config": {
            "temperature": provider.config.temperature,
            "top_p": provider.config.top_p,
            "max_tokens": provider.config.max_tokens,
            "seed": provider.config.seed,
        },
        **backend_info(),
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_generations(path: str | Path) -> List[dict]:
    recs = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                recs.append(json.loads(line))
    return recs
