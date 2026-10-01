"""Provedores de LLM do EASSE-PT."""

from __future__ import annotations

from typing import Optional

from easse_pt.llm.base import GenerationConfig, LLMProvider, LLMResponse
from easse_pt.llm.dummy import DummyProvider
from easse_pt.llm.ollama import OllamaProvider

PROVIDERS = {
    "ollama": OllamaProvider,
    "dummy": DummyProvider,
}


def get_provider(
    spec: str, config: Optional[GenerationConfig] = None, **kwargs
) -> LLMProvider:
    """Cria um provedor a partir de uma spec 'provider/model'.

    Exemplos: 'ollama/llama3.1:8b', 'ollama/qwen2.5:7b', 'dummy/rule-based'.
    Sem prefixo, assume-se 'ollama/'.
    """
    if "/" in spec:
        provider, model = spec.split("/", 1)
    else:
        provider, model = "ollama", spec
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"provedor desconhecido: {provider!r} (disponiveis: {list(PROVIDERS)})")
    return PROVIDERS[provider](model=model, config=config, **kwargs)


__all__ = [
    "LLMProvider", "LLMResponse", "GenerationConfig",
    "OllamaProvider", "DummyProvider", "get_provider", "PROVIDERS",
]
