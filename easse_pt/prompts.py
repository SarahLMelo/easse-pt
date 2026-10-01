"""Estrategias de prompting e pos-processamento das respostas.

Duas estrategias, conforme o plano de pesquisa:

  zero_shot : instrucao direta, uma unica etapa
  cot       : decomposicao guiada em tres passos (Zhang et al., 2026) —
              (1) identificar vocabulario complexo, (2) dividir oracoes
              subordinadas longas, (3) reescrever mantendo os fatos

Cada estrategia existe em dois niveis de simplificacao, espelhando as
referencias do PorSimples:

  natural : leitores em geral
  forte   : leitores com baixo nivel de alfabetizacao

O CoT deixa o raciocinio visivel, o que e desejavel para a analise
qualitativa, mas exige extrair a sentenca final — funcao de
`postprocess()`, que reconhece o marcador SIMPLIFICADA: e, na falta dele,
usa heuristicas de fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

STRATEGIES = ("zero_shot", "cot")
LEVELS = ("natural", "forte")

FINAL_MARKER = "SIMPLIFICADA:"

SYSTEM = (
    "Voce e um especialista em simplificacao textual em Portugues do Brasil. "
    "Voce reescreve sentencas para torna-las mais faceis de ler, sem inventar "
    "informacoes e sem omitir fatos essenciais. Responda sempre em Portugues do Brasil."
)

_AUDIENCE = {
    "natural": "um leitor adulto brasileiro em geral, sem formacao especializada",
    "forte": "um leitor com baixo nivel de alfabetizacao, como uma crianca de 10 anos",
}

# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
ZERO_SHOT = """\
Simplifique a seguinte sentenca em Portugues para que {audience} compreenda.

Regras:
- Preserve todos os fatos da sentenca original; nao acrescente informacao nova.
- Use palavras comuns e sentencas curtas.
- Voce pode dividir a sentenca em mais de uma sentenca.
- Responda APENAS com a versao simplificada, sem comentarios ou explicacoes.

Sentenca: {sentence}

{marker}"""

COT = """\
Simplifique a sentenca abaixo em Portugues para que {audience} compreenda.
Faca isso passo a passo.

Passo 1 - Vocabulario: liste as palavras ou expressoes dificeis da sentenca e,
para cada uma, escreva uma alternativa mais simples e de mesmo sentido.

Passo 2 - Estrutura: identifique oracoes subordinadas, apostos e encaixes longos,
e indique em quantas sentencas curtas o conteudo pode ser dividido.

Passo 3 - Reescrita: escreva a versao simplificada usando as decisoes dos passos
anteriores. Preserve TODOS os fatos da sentenca original e nao invente nada.

Escreva os passos e, na ultima linha, escreva a versao final precedida de
"{marker}" (apenas a sentenca simplificada, sem aspas e sem comentarios).

Sentenca: {sentence}
"""


@dataclass
class Prompt:
    text: str
    system: str
    strategy: str
    level: str


def build_prompt(sentence: str, strategy: str = "zero_shot", level: str = "forte") -> Prompt:
    if strategy not in STRATEGIES:
        raise ValueError(f"estrategia desconhecida: {strategy!r} (use {STRATEGIES})")
    if level not in LEVELS:
        raise ValueError(f"nivel desconhecido: {level!r} (use {LEVELS})")

    template = ZERO_SHOT if strategy == "zero_shot" else COT
    text = template.format(
        audience=_AUDIENCE[level], sentence=sentence.strip(), marker=FINAL_MARKER
    )
    return Prompt(text=text, system=SYSTEM, strategy=strategy, level=level)


# --------------------------------------------------------------------------- #
# Pos-processamento
# --------------------------------------------------------------------------- #
_PREFIX_NOISE = re.compile(
    r"^\s*(?:vers[ãa]o\s+simplificada|senten[çc]a\s+simplificada|simplifica[çc][ãa]o"
    r"|resposta|sa[íi]da|passo\s*3\s*-?\s*reescrita)\s*[:\-]\s*",
    re.IGNORECASE,
)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_MD_ONLY = re.compile(r"^[\s*_#>•\-]+$")
_MD_EDGE_START = re.compile(r"^[\s*_#>•\-]+")
_MD_EDGE_END = re.compile(r"(?<=[.!?])[\s*_]+$")
_FENCE = re.compile(r"^```[a-z]*\s*|\s*```$", re.MULTILINE)


def postprocess(raw: str, strategy: str = "zero_shot") -> str:
    """Extrai a sentenca simplificada final da resposta do modelo."""
    text = _THINK_BLOCK.sub(" ", raw or "")   # modelos "reasoning" (ex.: deepseek-r1)
    text = _FENCE.sub("", text).strip()
    if not text:
        return ""

    # 1) marcador explicito — o caminho esperado no CoT
    idx = text.upper().rfind(FINAL_MARKER)
    if idx != -1:
        text = text[idx + len(FINAL_MARKER):].strip()
    elif strategy == "cot":
        # 2) fallback: ultimo bloco nao vazio, descartando listas dos passos
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        if blocks:
            text = blocks[-1]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # linhas que sao so marcacao markdown (ex.: o "**" que sobra de "**Simplificada:**")
    lines = [ln for ln in lines if not _MD_ONLY.match(ln)]
    lines = [ln for ln in lines if not re.match(r"^(passo\s*\d|\d[\.\)]\s*(vocabul|estrut))",
                                                ln, re.IGNORECASE)]
    if not lines:
        return ""

    # respostas em zero-shot as vezes vem com uma frase de cortesia antes
    if strategy == "zero_shot" and len(lines) > 1 and re.search(
        r"(aqui est|segue|claro|abaixo)", lines[0], re.IGNORECASE
    ):
        lines = lines[1:]

    out = " ".join(lines)
    out = _PREFIX_NOISE.sub("", out).strip()
    out = out.strip('"').strip("'").strip()
    out = _MD_EDGE_START.sub("", out)
    out = _MD_EDGE_END.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


def all_configs(strategies: List[str] = None, levels: List[str] = None) -> List[Dict[str, str]]:
    """Produto cartesiano estrategia x nivel — usado pelo runner do benchmark."""
    strategies = strategies or list(STRATEGIES)
    levels = levels or list(LEVELS)
    return [{"strategy": s, "level": lv} for s in strategies for lv in levels]
