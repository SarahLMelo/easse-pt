"""Provedor Ollama — inferencia local, sem custo de API.

Requer o Ollama rodando na maquina (padrao: http://localhost:11434).
Instalacao no Windows: https://ollama.com/download  — depois, no terminal:

    ollama pull llama3.1:8b
    ollama pull qwen2.5:7b
    ollama pull deepseek-r1:7b
    ollama serve            # normalmente ja sobe sozinho como servico

Usa a rota /api/generate (prompt unico) com `stream=false`. A temperatura
padrao e 0 e a `seed` e fixa, para que a mesma sentenca produza a mesma
simplificacao entre execucoes — requisito do benchmark reprodutivel.
"""

from __future__ import annotations

import time
from typing import List, Optional

import requests

from easse_pt.llm.base import GenerationConfig, LLMProvider, LLMResponse

DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1:8b",
        config: Optional[GenerationConfig] = None,
        host: str = DEFAULT_HOST,
        timeout: int = 300,
        keep_alive: str = "10m",
    ):
        super().__init__(model, config)
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
                **({"seed": self.config.seed} if self.config.seed is not None else {}),
                **({"stop": self.config.stop} if self.config.stop else {}),
                **self.config.extra,
            },
        }
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(
            text=(data.get("response") or "").strip(),
            raw=data,
            latency_s=time.perf_counter() - t0,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    # ------------------------------------------------------------------ #
    def list_models(self) -> List[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=15)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def health_check(self) -> bool:
        try:
            models = self.list_models()
        except Exception:
            return False
        if self.model in models:
            return True
        # tolera "llama3.1" quando o instalado e "llama3.1:8b" e vice-versa
        base = self.model.split(":")[0]
        return any(m.split(":")[0] == base for m in models)

    def ensure_model(self) -> None:
        """Mensagem de erro util quando o modelo nao foi baixado."""
        try:
            models = self.list_models()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama nao respondeu em {self.host}. Ele esta rodando? "
                "(instale em https://ollama.com e rode `ollama serve`)"
            ) from exc
        if not self.health_check():
            raise RuntimeError(
                f"modelo '{self.model}' nao encontrado no Ollama. "
                f"Rode `ollama pull {self.model}`. Instalados: {', '.join(models) or '(nenhum)'}"
            )
