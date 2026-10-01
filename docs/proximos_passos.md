# Próximos passos — do ambiente pronto ao benchmark

O ambiente cobre a infraestrutura do **Mês 1** do cronograma. O que falta é
principalmente dado e decisão de pesquisa, não código.

## Mês 1 — Preparação de infraestrutura e dados

- [x] Estrutura do repositório, instalação reprodutível, testes
- [x] Extensão PT do EASSE: tokenizador PT-BR + Flesch-Brasil no lugar do FKGL
- [x] Formato EASSE de saída (`orig_sents`, `refs_sents` em dois níveis)
- [x] Conversores para os formatos em que o PorSimples circula
- [ ] **Baixar o PorSimples/PorSimpleFul** e rodar `easse-pt corpus-build data/raw`
- [ ] Conferir o alinhamento sentença a sentença depois da conversão
      (`easse-pt corpus-stats`: o Flesch de `forte` deve ser maior que o de
      `natural`, que deve ser maior que o do original — se não for, o
      alinhamento saiu trocado)
- [x] Instalar o Ollama e baixar o `llama3.1:8b` (outros: `ollama pull qwen2.5:7b`)
- [x] Instalar a camada semântica (`requirements-semantic.txt`) e o spaCy PT

### Decisões de pesquisa a tomar antes do Mês 2

1. **Recorte do corpus.** O PorSimpleFul inteiro pode ser grande demais para
   `modelos × estratégias × níveis` em notebook. Defina um recorte fixo
   (ex.: 300 sentenças por domínio, amostradas com semente fixa via
   `Dataset.subset`) e registre-o na monografia.
2. **Lista final de modelos.** O plano cita Llama 3, DeepSeek e GPT-4o. GPT-4o
   é proprietário e exige API paga — se ficar de fora, diga isso explicitamente
   na metodologia. `configs/benchmark.yaml` tem a grade aberta a comentar.
   Para adicionar um provedor de API, basta implementar `LLMProvider` em
   `easse_pt/llm/` e registrá-lo em `PROVIDERS`.
3. **Tratamento de respostas longas.** Modelos *reasoning* (deepseek-r1) gastam
   muitos tokens antes da resposta. `postprocess()` já remove blocos `<think>`,
   mas vale medir o tempo por sentença no piloto antes da rodada completa.

## Mês 2 — Execução do benchmark

```bash
easse-pt benchmark --config configs/benchmark-piloto.yaml   # calibra prompts, mede tempo
easse-pt benchmark --config configs/benchmark.yaml          # rodada completa
```

As runs são retomáveis: interromper e repetir o comando só gera o que falta.

Antes da rodada completa, leia ~20 saídas do piloto à mão. Os prompts em
`easse_pt/prompts.py` são o principal ponto de calibração — sobretudo o
marcador `SIMPLIFICADA:`, de que depende a extração da resposta no CoT.

## Mês 3 — Análise, redação e defesa

```bash
easse-pt report
python scripts/analise_erros.py "data/outputs/*.per_sent.csv"
```

- `benchmark.csv` / `benchmark.md`: tabela comparativa, já com a linha
  `[referencia humana]` como teto.
- `<run>.per_sent.csv`: base da análise qualitativa — ordene por SARI crescente
  para achar os piores casos.
- `<run>.jsonl`: saída crua, com o raciocínio em três passos do CoT.

Perguntas que o material já responde e que costumam virar seção de resultados:

- CoT supera zero-shot? Em qual nível (natural/forte) a diferença é maior?
- O ganho de legibilidade vem acompanhado de perda de significado?
  (cruzar `ΔFlesch` com `Sem.(orig)` — é a tensão apontada por Zhang et al.)
- Quais modelos apenas copiam a entrada? (`Copia %` alto, `add` baixo)
- Que distância separa a melhor LLM da referência humana?

## Entregáveis do plano

| Entregável | Estado |
|---|---|
| Módulo open-source EASSE-PT | pronto — falta publicar o repositório |
| Dataset formatado (JSON/texto alinhado do PorSimples) | pipeline pronto; depende do download |
| Relatório de benchmark | gerado por `easse-pt report` após as runs |
