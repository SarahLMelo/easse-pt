"""Pipelines de geracao, avaliacao e relatorio."""

from easse_pt.pipeline.evaluate import evaluate_references, evaluate_run
from easse_pt.pipeline.generate import RunSpec, load_generations, output_path, run_generation
from easse_pt.pipeline.report import collect_scores, to_markdown, write_report

__all__ = [
    "run_generation", "load_generations", "output_path", "RunSpec",
    "evaluate_run", "evaluate_references",
    "write_report", "collect_scores", "to_markdown",
]
