"""CLI do EASSE-PT (argparse — sem dependencias externas).

    easse-pt doctor                       # diagnostico do ambiente
    easse-pt corpus-sample                # materializa a amostra sintetica
    easse-pt corpus-build data/raw        # converte o PorSimples para o formato EASSE
    easse-pt corpus-stats                 # estatisticas e legibilidade do corpus
    easse-pt generate --model ollama/llama3.1:8b --strategy cot --level forte
    easse-pt evaluate data/outputs/<run>.jsonl
    easse-pt benchmark --config configs/benchmark.yaml
    easse-pt report

Equivalente sem instalar o pacote:  python -m easse_pt.cli <comando>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from easse_pt import __version__

DEFAULT_DATASET = "data/processed/porsimplesent.jsonl"
DEFAULT_OUT = "data/outputs"


# --------------------------------------------------------------------------- #
# Utilitarios de saida
# --------------------------------------------------------------------------- #
def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    line = "+".join("-" * (w + 2) for w in widths)
    print(f"\n{title}")
    print(line)
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for r in rows:
        print(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    print(line)


def _load_dataset(path: Optional[str]):
    from easse_pt.corpus import Dataset, load_sample

    if path is None:
        p = Path(DEFAULT_DATASET)
        if p.exists():
            return Dataset.from_jsonl(p)
        print("[aviso] Corpus do PorSimples nao encontrado em "
              f"{DEFAULT_DATASET} — usando a amostra sintetica "
              "(serve para validar o pipeline, nao para o benchmark do TCC).",
              file=sys.stderr)
        return load_sample()
    if path in ("sample", "amostra"):
        return load_sample()
    return Dataset.from_jsonl(path)


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    import importlib

    from easse_pt.tokenization import backend_info

    rows: List[List[str]] = []

    def add(name: str, ok: bool, detail) -> None:
        rows.append([name, "ok" if ok else "FALTA", str(detail)])

    def mod(name: str, label: str):
        try:
            m = importlib.import_module(name)
            add(label, True, getattr(m, "__version__", "instalado"))
        except Exception:
            add(label, False, "nao instalado")

    add("Python", sys.version_info >= (3, 9), sys.version.split()[0])
    for name, label in [
        ("numpy", "numpy"), ("pandas", "pandas"), ("yaml", "pyyaml"),
        ("requests", "requests"), ("sacrebleu", "sacrebleu (BLEU canonico)"),
        ("pyphen", "pyphen (silabas PT)"), ("tqdm", "tqdm (opcional)"),
        ("spacy", "spacy (opcional)"), ("nltk", "nltk (opcional)"),
        ("sentence_transformers", "sentence-transformers (opcional)"),
        ("bert_score", "bert-score (opcional)"),
    ]:
        mod(name, label)

    info = backend_info()
    add("Tokenizador PT", True, f"{info['tokenizer_backend']} ({info['tokenizer_model']})")

    try:
        from easse_pt.metrics.bleu import _has_sacrebleu

        add("Motor de BLEU", True, "sacrebleu" if _has_sacrebleu() else "interno (fallback)")
    except Exception as exc:
        add("Motor de BLEU", False, exc)

    try:
        from easse_pt.llm import OllamaProvider

        models = OllamaProvider(host=args.ollama_host).list_models()
        add("Ollama", bool(models), ", ".join(models) if models else "rodando, sem modelos baixados")
    except Exception:
        add("Ollama", False, f"sem resposta em {args.ollama_host} — rode `ollama serve`")

    p = Path(DEFAULT_DATASET)
    if p.exists():
        from easse_pt.corpus import Dataset

        ds = Dataset.from_jsonl(p)
        add("Corpus PorSimples", True, f"{len(ds)} sentencas, niveis {ds.available_levels()}")
    else:
        add("Corpus PorSimples", False, f"{DEFAULT_DATASET} ausente — `easse-pt corpus-build data/raw`")

    _print_table(f"EASSE-PT {__version__} — diagnostico do ambiente",
                 ["Componente", "Status", "Detalhe"], rows)
    faltando = [r[0] for r in rows if r[1] == "FALTA" and "opcional" not in r[0]]
    if faltando:
        print(f"\nPendencias: {', '.join(faltando)}")
    return 0


def cmd_corpus_sample(args) -> int:
    from easse_pt.corpus import load_sample

    ds = load_sample()
    path = ds.to_jsonl(args.out)
    print(f"{len(ds)} sentencas -> {path}")
    for suf, p in ds.to_easse_files(Path(args.out).parent).items():
        print(f"  {suf}: {p}")
    return 0


def cmd_corpus_build(args) -> int:
    from easse_pt.corpus import DOWNLOAD_NOTE, build_dataset

    try:
        ds = build_dataset(args.raw_dir, name=args.name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"erro: {exc}\n", file=sys.stderr)
        print(DOWNLOAD_NOTE, file=sys.stderr)
        return 1

    path = ds.to_jsonl(args.out)
    print(f"{len(ds)} sentencas -> {path}")
    print(f"Niveis completos: {ds.available_levels()}")
    for suf, p in ds.to_easse_files(Path(args.out).parent).items():
        print(f"  {suf}: {p}")
    return 0


def cmd_corpus_stats(args) -> int:
    from easse_pt.metrics.readability import flesch_brasil_level

    ds = _load_dataset(args.dataset)
    rows = [
        [k, f"{v:.2f}", flesch_brasil_level(v) if k.startswith("flesch") else ""]
        for k, v in ds.stats().items()
    ]
    _print_table(f"Corpus: {ds.name}", ["Metrica", "Valor", "Faixa"], rows)
    return 0


def cmd_generate(args) -> int:
    from easse_pt.llm import GenerationConfig, get_provider
    from easse_pt.pipeline import run_generation

    ds = _load_dataset(args.dataset)
    cfg = GenerationConfig(temperature=args.temperature, max_tokens=args.max_tokens)
    provider = get_provider(args.model, config=cfg)
    if hasattr(provider, "ensure_model"):
        try:
            provider.ensure_model()
        except RuntimeError as exc:
            print(f"erro: {exc}", file=sys.stderr)
            return 1

    path = run_generation(ds, provider, strategy=args.strategy, level=args.level,
                          out_dir=args.out_dir, resume=not args.no_resume, limit=args.limit)
    print(f"geracoes -> {path}")
    return 0


def cmd_evaluate(args) -> int:
    from easse_pt.pipeline import evaluate_run

    ds = _load_dataset(args.dataset)
    scores = evaluate_run(args.generations, ds, semantic=not args.no_semantic)
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark(args) -> int:
    import yaml

    from easse_pt.llm import GenerationConfig, get_provider
    from easse_pt.pipeline import evaluate_references, evaluate_run, run_generation, write_report

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    models: List[str] = cfg.get("models", [])
    strategies: List[str] = cfg.get("strategies", ["zero_shot", "cot"])
    levels: List[str] = cfg.get("levels", ["natural", "forte"])
    gen_cfg = GenerationConfig(**(cfg.get("generation") or {}))
    limit = args.limit if args.limit is not None else cfg.get("limit")

    ds = _load_dataset(args.dataset)
    print(f"Corpus: {ds.name} ({len(ds)} sentencas)")
    print(f"Grade: {len(models)} modelos x {len(strategies)} estrategias x {len(levels)} niveis")

    for model in models:
        provider = get_provider(model, config=gen_cfg)
        if hasattr(provider, "ensure_model"):
            try:
                provider.ensure_model()
            except RuntimeError as exc:
                print(f"[pulando {model}] {exc}", file=sys.stderr)
                continue
        for strategy in strategies:
            for level in levels:
                print(f"\n>> {model} | {strategy} | {level}")
                path = run_generation(ds, provider, strategy=strategy, level=level,
                                      out_dir=args.out_dir, limit=limit)
                sc = evaluate_run(path, ds, semantic=not args.no_semantic)
                print(f"   SARI {sc['sari']:.2f} | BLEU {sc['bleu']:.2f} "
                      f"| Flesch {sc['flesch_sys']:.1f} (Δ {sc['flesch_gain']:+.1f})")

    refs = list(evaluate_references(ds.subset(limit) if limit else ds).values())
    paths = write_report(args.out_dir, args.out_dir, reference_rows=refs)
    print(f"\nrelatorio -> {paths['markdown']}  |  {paths['csv']}")
    return 0


def cmd_report(args) -> int:
    from easse_pt.pipeline import evaluate_references, write_report

    refs = None
    if not args.no_references:
        refs = list(evaluate_references(_load_dataset(args.dataset)).values())
    paths = write_report(args.out_dir, args.out_dir, reference_rows=refs)
    print(Path(paths["markdown"]).read_text(encoding="utf-8"))
    print(f"-> {paths['csv']}\n-> {paths['markdown']}")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="easse-pt",
        description="EASSE-PT — avaliacao automatica de simplificacao de sentencas em PT-BR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"easse-pt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="diagnostico do ambiente")
    d.add_argument("--ollama-host", default="http://localhost:11434")
    d.set_defaults(func=cmd_doctor)

    cs = sub.add_parser("corpus-sample", help="materializa a amostra sintetica de 20 sentencas")
    cs.add_argument("--out", default="data/processed/amostra_sintetica.jsonl")
    cs.set_defaults(func=cmd_corpus_sample)

    cb = sub.add_parser("corpus-build", help="converte o PorSimples bruto para o formato EASSE")
    cb.add_argument("raw_dir")
    cb.add_argument("--out", default=DEFAULT_DATASET)
    cb.add_argument("--name", default="porsimplesent")
    cb.set_defaults(func=cmd_corpus_build)

    ct = sub.add_parser("corpus-stats", help="tamanho e legibilidade de cada nivel")
    ct.add_argument("--dataset", default=None)
    ct.set_defaults(func=cmd_corpus_stats)

    g = sub.add_parser("generate", help="gera simplificacoes com uma LLM")
    g.add_argument("--model", default="ollama/llama3.1:8b")
    g.add_argument("--strategy", default="zero_shot", choices=["zero_shot", "cot"])
    g.add_argument("--level", default="forte", choices=["natural", "forte"])
    g.add_argument("--dataset", default=None)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--out-dir", default=DEFAULT_OUT)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--max-tokens", type=int, default=512)
    g.add_argument("--no-resume", action="store_true")
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("evaluate", help="calcula as metricas de uma run")
    e.add_argument("generations")
    e.add_argument("--dataset", default=None)
    e.add_argument("--no-semantic", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="roda a grade completa e monta o relatorio")
    b.add_argument("--config", default="configs/benchmark.yaml")
    b.add_argument("--dataset", default=None)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--out-dir", default=DEFAULT_OUT)
    b.add_argument("--no-semantic", action="store_true")
    b.set_defaults(func=cmd_benchmark)

    r = sub.add_parser("report", help="reconsolida a tabela do benchmark")
    r.add_argument("--out-dir", default=DEFAULT_OUT)
    r.add_argument("--dataset", default=None)
    r.add_argument("--no-references", action="store_true")
    r.set_defaults(func=cmd_report)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
