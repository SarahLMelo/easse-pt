# EASSE-PT

Adaptação do framework **EASSE** (Alva-Manchego et al., 2019) para o **Português do Brasil** e
ambiente de benchmark de LLMs em simplificação de sentenças, conforme o plano de pesquisa do TCC.

O que o repositório entrega:

| Objetivo do plano | Onde está |
|---|---|
| 1. Tratamento e estruturação de dados (PorSimples → formato EASSE, níveis Natural e Forte) | `easse_pt/corpus/` · `easse-pt corpus-build` |
| 2. Adaptação do EASSE (tokenizador PT-BR, Flesch-Brasil no lugar do FKGL, preservação semântica) | `easse_pt/tokenization.py` · `easse_pt/metrics/` |
| 3. Execução do benchmark (LLMs abertas, prompting direto vs. passo a passo/CoT) | `easse_pt/llm/` · `easse_pt/prompts.py` · `easse-pt benchmark` |
| 4. Análise de desempenho e erros | `easse-pt evaluate` (CSV por sentença) · `easse-pt report` |

---

## 1. Instalação

Pré-requisito: **Python 3.9+** (recomendado 3.11).

### Windows (PowerShell)

O ambiente já está instalado nesta máquina (`.venv` com Python 3.12, Ollama com
`llama3.1:8b`). As versões exatas de cada pacote estão registradas em
`docs/ambiente.txt` — cite-as na metodologia. Para usar:

```powershell
cd $HOME\Documents\easse-pt
.\.venv\Scripts\Activate.ps1
easse-pt doctor
```

Para recriar o ambiente do zero (apague a pasta `.venv` antes):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Semantic -Spacy
```

### Linux / macOS

```bash
cd easse-pt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
easse-pt doctor
```

`easse-pt doctor` imprime uma tabela com o estado de cada componente (dependências,
tokenizador ativo, motor de BLEU, Ollama, corpus). Use-o sempre que algo falhar.

### Camadas opcionais

```bash
# Preservação semântica (BERTScore-PT / sentence-transformers) — objetivo 4 do plano
pip install -r requirements-semantic.txt

# Tokenização linguística mais precisa (recomendado para os números finais do TCC)
pip install spacy
python -m spacy download pt_core_news_sm
```

O pacote **roda sem nenhuma das camadas opcionais**: há fallback interno para BLEU (implementação
própria de BLEU-4) e para a contagem de sílabas (heurística de grupos vocálicos). Os fallbacks são
registrados nos resultados (campos `bleu_engine` e `tokenizer_backend`) — reporte-os na monografia,
pois números gerados com backends diferentes não são estritamente comparáveis.

### LLMs locais (Ollama)

```bash
# instale em https://ollama.com/download
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
# opcional, respostas longas e lentas:
ollama pull deepseek-r1:7b
ollama list          # confere o que está baixado
```

O Ollama sobe sozinho como serviço em `http://localhost:11434`. Se não, rode `ollama serve`.

---

## 2. Corpus

O **PorSimples / PorSimpleFul** é distribuído pelo NILC (ICMC/USP) e **não** é redistribuído aqui.
Baixe pelos links do plano de pesquisa, extraia em `data/raw/` e converta:

```bash
easse-pt corpus-build data/raw --out data/processed/porsimplesent.jsonl
easse-pt corpus-stats
```

O conversor reconhece três formatos: TSV/CSV alinhado por sentença (PorSimplesSent), trios XML do
NILC (`*ori*.xml` / `_nat` / `_str`) e arquivos-texto paralelos `.ori` / `.nat` / `.str`. Nomes de
coluna são detectados com tolerância (`original`, `ori`, `complex`, `natural`, `nat`, `forte`,
`strong`, ...).

Saída: um **JSONL** (uma linha por sentença) e os **arquivos paralelos no padrão EASSE**
(`<nome>.orig`, `<nome>.natural`, `<nome>.forte`), que o EASSE original consome diretamente.

### Amostra sintética

Enquanto o corpus não chega, há 20 sentenças sintéticas (escritas para este repositório, **não**
do PorSimples) com os dois níveis de referência, só para validar o pipeline:

```bash
easse-pt corpus-sample
easse-pt corpus-stats --dataset sample
```

Os números do TCC têm de sair do PorSimpleFul; a amostra serve apenas para checar encanamento.

---

## 3. Uso

### Rodada piloto (calibrar prompts e medir tempo por sentença)

```bash
easse-pt generate --model ollama/llama3.1:8b --strategy cot --level forte --limit 20
easse-pt evaluate data/outputs/porsimplesent__ollama_llama3.1-8b__cot__forte.jsonl
```

### Benchmark completo

```bash
easse-pt benchmark --config configs/benchmark-piloto.yaml   # 1 modelo, 20 sentenças
easse-pt benchmark --config configs/benchmark.yaml          # grade completa
easse-pt report                                             # reconsolida a tabela
```

A grade é `modelos × estratégias × níveis`, definida em `configs/benchmark.yaml`. Cada combinação
é uma *run* independente e **retomável**: se a execução for interrompida, rodar o mesmo comando
gera apenas as sentenças que faltam.

### Artefatos por run (em `data/outputs/`)

| Arquivo | Conteúdo |
|---|---|
| `<run>.jsonl` | saída crua do modelo + versão pós-processada, por sentença |
| `<run>.meta.json` | modelo, parâmetros de geração, tokenizador, data, nº de erros |
| `<run>.scores.json` | métricas agregadas (a linha da run no benchmark) |
| `<run>.per_sent.csv` | métricas por sentença — base da análise qualitativa de erros |
| `benchmark.csv` / `benchmark.md` | tabela comparativa de todas as runs |

A saída **crua** é preservada de propósito: no CoT, o raciocínio em três passos é material da
análise qualitativa do Mês 3.

---

## 4. O que foi adaptado do EASSE

| Componente | EASSE original | EASSE-PT |
|---|---|---|
| Tokenização | tokenizadores do inglês (13a/Moses) | spaCy `pt_core_news_sm` → NLTK `punkt/pt` → regex PT (fallback), com clise (`diz-se`), contrações e abreviaturas brasileiras |
| Legibilidade | FKGL (calibrado para o inglês) | **Flesch-Brasil** (Martins et al., 1996): `248.835 − 1.015·ASL − 84.6·ASW`, com silabação por dicionário pt_BR |
| SARI / BLEU | n-gramas sobre tokenização inglesa | mesma fórmula, n-gramas sobre a tokenização PT-BR |
| Preservação semântica | — | **BERTScore-PT** (BERTimbau) contra as referências + similaridade de embeddings contra a sentença original |
| Referências | 1..N referências | dois níveis do PorSimples: **Natural** e **Forte** |

### Como ler as métricas

- **SARI** é a métrica principal (maior = melhor). As parcelas `keep` / `del` / `add` mostram *qual*
  operação de edição o modelo acerta — `add` costuma ser a mais baixa e é onde as LLMs se separam.
- **BLEU** mede proximidade das referências e **penaliza reescritas ousadas**; serve de contrapeso,
  não de critério de qualidade.
- **Flesch-Brasil**: maior = mais fácil. `ΔFlesch` é o ganho sobre a sentença original.
- **BERTScore-PT** e `Sem.(orig)` medem preservação de significado — contra as referências e contra
  a sentença fonte, respectivamente.
- `Copia %` alta com `ΔFlesch` baixo denuncia modelo que apenas repete a entrada. É o modo de falha
  mais comum em zero-shot, e o motivo de o relatório incluir a linha **`[referencia humana]`**: ela
  é o teto contra o qual as notas das LLMs devem ser lidas.

---

## 5. Estrutura

```
easse_pt/
  tokenization.py     tokenizador PT-BR (3 backends, com fallback)
  prompts.py          zero-shot e CoT (Zhang et al., 2026) x níveis natural/forte
  metrics/
    sari.py           SARI multi-referência
    bleu.py           BLEU (sacrebleu ou implementação interna)
    readability.py    Flesch-Brasil + contagem de sílabas
    semantic.py       BERTScore-PT e similaridade de embeddings
    quality.py        compressão, divisão de sentenças, cópia literal, Levenshtein
  corpus/
    dataset.py        estrutura orig/natural/forte, JSONL e export para o EASSE
    porsimples.py     conversores TSV/XML/texto paralelo
  llm/
    ollama.py         inferência local
    dummy.py          baseline por regras (valida o pipeline sem LLM)
  pipeline/
    generate.py       execução retomável das runs
    evaluate.py       métricas de corpus e por sentença + teto humano
    report.py         tabela comparativa (CSV/Markdown)
  cli.py              interface de linha de comando
configs/              grades do benchmark
data/raw|processed|outputs/
tests/                44 testes (pytest ou unittest)
docs/ambiente.txt     registro das versoes instaladas (reprodutibilidade)
```

## 6. Testes

```bash
pytest -q                       # se o pytest estiver instalado
python -m unittest discover -s tests   # sem nenhuma dependência extra
```

## 7. Reprodutibilidade

- `temperature = 0.0` e `seed` fixa por padrão: a mesma sentença produz a mesma simplificação.
- Cada run grava seus parâmetros em `<run>.meta.json`.
- O motor de BLEU e o backend do tokenizador entram nos resultados — registre-os na monografia.
- O corpus não é redistribuído; a conversão é reprodutível a partir do download oficial.

## Referências

- Alva-Manchego, Martin, Scarton, Specia (2019). *EASSE: Easier Automatic Sentence Simplification Evaluation.* EMNLP-IJCNLP.
- Xu, Napoles, Pavlick, Chen, Callison-Burch (2016). *Optimizing Statistical Machine Translation for Text Simplification.* TACL.
- Martins, Ghiraldelo, Nunes, Oliveira Jr. (1996). *Readability formulas applied to textbooks in Brazilian Portuguese.* ICMC/USP.
- Aluísio & Gasperin. *Fostering Digital Inclusion and Accessibility: the PorSimples project.*
- Scalercio, De Souza, Finatto, Paes (2025). *Evaluating LLMs for Portuguese Sentence Simplification with Linguistic Insights.* ACL.
- Zhang et al. (2026). *Let's Simplify Step by Step: Guiding LLM Towards Multilingual Unsupervised Proficiency-Controlled Sentence Simplification.* Findings of EACL.
- Martin et al. (2022). *MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases.* LREC.
