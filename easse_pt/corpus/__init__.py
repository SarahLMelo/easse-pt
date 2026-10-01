"""Camada de corpus do EASSE-PT."""

from pathlib import Path

from easse_pt.corpus.dataset import LEVELS, Dataset, Sample
from easse_pt.corpus.porsimples import DOWNLOAD_NOTE, build_dataset

SAMPLE_PATH = Path(__file__).parent / "data" / "amostra_sintetica.jsonl"


def load_sample() -> Dataset:
    """Amostra SINTETICA de 20 sentencas, escrita para este repositorio.

    Serve para validar o pipeline de ponta a ponta (geracao + metricas) sem
    depender do download do PorSimples. NAO deve ser usada como corpus de
    avaliacao no TCC: os numeros do benchmark tem de sair do PorSimpleFul.
    """
    return Dataset.from_jsonl(SAMPLE_PATH, name="amostra-sintetica")


__all__ = [
    "Dataset", "Sample", "LEVELS",
    "build_dataset", "load_sample", "SAMPLE_PATH", "DOWNLOAD_NOTE",
]
