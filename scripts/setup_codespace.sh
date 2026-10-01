#!/usr/bin/env bash
# Rode uma vez, no terminal do Codespace recem-criado:
#     bash scripts/setup_codespace.sh
# Monta dois ambientes:
#   1. EASSE-PT (este repositorio) no Python do container
#   2. EASSE original em /workspaces/easse-original, com um venv proprio
# Log completo em /workspaces/setup-codespace.log
set -uo pipefail
LOG=/workspaces/setup-codespace.log
exec > >(tee -a "$LOG") 2>&1
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ORIG=/workspaces/easse-original
falhas=()
etapa() { echo; echo "==> $1"; }
check() { if [ "$1" -ne 0 ]; then echo "    FALHOU: $2"; falhas+=("$2"); fi; }

etapa "EASSE-PT: dependencias"
python -m pip install --upgrade pip
pip install -r requirements.txt spacy;            check $? "dependencias EASSE-PT"
pip install -e .;                                 check $? "pacote easse-pt"
python -m spacy download pt_core_news_sm;         check $? "modelo spaCy pt_core_news_sm"

etapa "EASSE-PT: testes"
python -m pytest -q -p no:cacheprovider;          check $? "testes EASSE-PT"

etapa "EASSE original: codigo"
if [ ! -d "$ORIG/easse/.git" ]; then
  mkdir -p "$ORIG"
  git clone https://github.com/feralvam/easse.git "$ORIG/easse"; check $? "clone do EASSE"
fi

etapa "EASSE original: venv separado"
python -m venv "$ORIG/.venv"
VPIP="$ORIG/.venv/bin/pip"
"$VPIP" install --upgrade pip setuptools wheel
# torch so de CPU: evita baixar ~3 GB de bibliotecas CUDA
"$VPIP" install torch --index-url https://download.pytorch.org/whl/cpu; check $? "torch (CPU)"
"$VPIP" install -e "$ORIG/easse";                  check $? "EASSE original"

etapa "EASSE original: exemplo do README"
cd "$ORIG/easse" && "$ORIG/.venv/bin/easse" evaluate -t turkcorpus_test -m 'bleu,sari,fkgl' -q \
  < easse/resources/data/system_outputs/turkcorpus/test/ACCESS | tee "$ORIG/exemplo-readme.txt"
check "${PIPESTATUS[0]}" "exemplo do README"
cd "$REPO"

echo
if [ ${#falhas[@]} -eq 0 ]; then
  echo "SETUP DO CODESPACE CONCLUIDO SEM ERROS."
else
  echo "SETUP DO CODESPACE TERMINOU COM FALHAS: ${falhas[*]}"
fi
echo "EASSE original: source $ORIG/.venv/bin/activate"
