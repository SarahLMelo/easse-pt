"""Interface comum dos provedores de LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GenerationConfig:
    temperature: float = 0.0        # deterministico: o benchmark precisa ser reprodutivel
    top_p: float = 0.9
    max_tokens: int = 512
    seed: Optional[int] = 13
    stop: List[str] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    raw: Dict = field(default_factory=dict)
    latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMProvider(ABC):
    """Contrato minimo: dado um prompt (e um system opcional), devolver texto."""

    name: str = "base"

    def __init__(self, model: str, config: Optional[GenerationConfig] = None):
        self.model = model
        self.config = config or GenerationConfig()

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        ...

    def generate_batch(self, prompts: List[str], system: Optional[str] = None) -> List[LLMResponse]:
        return [self.generate(p, system=system) for p in prompts]

    def health_check(self) -> bool:
        try:
            return bool(self.generate("Responda apenas: ok").text)
        except Exception:
            return False

    @property
    def tag(self) -> str:
        """Identificador usado em nomes de arquivo e no relatorio."""
        return f"{self.name}:{self.model}".replace("/", "_").replace(":", "-")
