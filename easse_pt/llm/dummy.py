"""Provedor ficticio — testes e validacao do pipeline sem LLM.

Aplica transformacoes superficiais e deterministicas (troca de conectivos
formais, remocao de encaixes entre virgulas, quebra em oracoes). Nao e um
baseline de pesquisa: serve para checar que geracao, cache, avaliacao e
relatorio funcionam de ponta a ponta antes de gastar GPU com os modelos reais.
"""

from __future__ import annotations

import re
from typing import Optional

from easse_pt.llm.base import LLMProvider, LLMResponse

_SUBS = [
    (r"\bem\s+raz[ãa]o\s+d[eoa]s?\b", "por causa de"),
    (r"\bdevido\s+[àa]s?\b", "por causa de"),
    (r"\bapesar\s+d[eoa]s?\b", "mesmo com"),
    (r"\bembora\b", "mesmo que"),
    (r"\bentretanto\b", "mas"),
    (r"\bcontudo\b", "mas"),
    (r"\bno\s+entanto\b", "mas"),
    (r"\bportanto\b", "por isso"),
    (r"\bsegundo\b", "de acordo com"),
    (r"\bcerca\s+de\b", "quase"),
    (r"\brealizad[ao]s?\b", "feito"),
    (r"\bsignificativamente\b", "muito"),
    (r"\bpermanecer(am|a|ão)\b", "ficaram"),
    (r"\bresid[êe]ncias?\b", "casa"),
    (r"\bestabelecimentos?\s+comerciais?\b", "lojas"),
]


class DummyProvider(LLMProvider):
    name = "dummy"

    def __init__(self, model: str = "rule-based", config=None):
        super().__init__(model, config)

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        sent = _extract_sentence(prompt)
        out = sent
        for pat, rep in _SUBS:
            out = re.sub(pat, rep, out, flags=re.IGNORECASE)
        # quebra encaixes longos entre virgulas em uma segunda sentenca
        m = re.match(r"^(.{15,}?),\s+(que|o que|a qual|os quais)\s+(.+)$", out)
        if m:
            head = m.group(1).rstrip(".")
            tail = m.group(3).rstrip(".")
            out = f"{head}. Isso {tail}."
        out = re.sub(r"\s+", " ", out).strip()
        if not out.endswith((".", "!", "?")):
            out += "."
        return LLMResponse(text=out, raw={"provider": "dummy"}, latency_s=0.0)

    def health_check(self) -> bool:
        return True


def _extract_sentence(prompt: str) -> str:
    """Recupera a sentenca de entrada do prompt (marcada por 'Sentenca:')."""
    m = re.search(r"Senten[çc]a(?:\s+original)?:\s*(.+?)(?:\n\s*\n|\nSimplifica|\Z)", prompt,
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().strip('"')
    return prompt.strip().splitlines()[-1].strip()
