# Preparacao do ambiente EASSE-PT no Windows (PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -Semantic -Spacy
#
# Opcoes: -Semantic        instala torch/transformers/BERTScore-PT (~2 GB)
#         -Spacy           instala spaCy + modelo pt_core_news_sm
#         -Python <exe>    caminho do python.exe a usar (padrao: 3.12 > 3.11 > 3.10)
#
# Tudo o que aparece na tela tambem vai para setup.log, na raiz do projeto.
# ATENCAO: manter este arquivo em ASCII puro (o PowerShell 5.1 le UTF-8 sem BOM
# como ANSI, e caracteres como travessao quebram a leitura do script).

param(
    [switch]$Semantic,
    [switch]$Spacy,
    [string]$Python = ""
)

$ErrorActionPreference = "Continue"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$log = Join-Path $root "setup.log"
Start-Transcript -Path $log -Force | Out-Null

$falhas = @()

function Etapa($titulo, $bloco) {
    Write-Host ""
    Write-Host "==> $titulo" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $bloco
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    FALHOU (codigo $LASTEXITCODE)" -ForegroundColor Red
        $script:falhas += $titulo
    }
}

function Find-Python {
    # Prefere versoes com wheels garantidas de torch/spaCy.
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in "3.12", "3.11", "3.10") {
            $exe = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
        }
    }
    foreach ($v in "312", "311", "310") {
        $cand = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$v\python.exe"
        if (Test-Path $cand) { return $cand }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $exe = & python -c "import sys; print(sys.executable) if (3,10) <= sys.version_info[:2] <= (3,12) else sys.exit(1)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    }
    return $null
}

Write-Host "EASSE-PT - instalacao local" -ForegroundColor Green
Write-Host "Pasta: $root"
Write-Host "Data : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# --------------------------------------------------------------------------- #
if (-not $Python) { $Python = Find-Python }
if (-not $Python -or -not (Test-Path $Python)) {
    Write-Host ""
    Write-Host "ERRO: Python 3.10-3.12 nao encontrado." -ForegroundColor Red
    Write-Host "Instale o Python 3.12 (winget install -e --id Python.Python.3.12) ou em" -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/ marcando 'Add python.exe to PATH'." -ForegroundColor Red
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "Python: $Python"
& $Python --version

Etapa "Criando o ambiente virtual (.venv)" {
    if (Test-Path ".venv\Scripts\python.exe") {
        Write-Host "    .venv ja existe - reaproveitando"
    } else {
        & $Python -m venv .venv
    }
}

$pip = Join-Path $root ".venv\Scripts\pip.exe"
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Host "ERRO: o ambiente virtual nao foi criado." -ForegroundColor Red
    Stop-Transcript | Out-Null
    exit 1
}

Etapa "Atualizando o pip" { & $venvPy -m pip install --upgrade pip }
Etapa "Instalando as dependencias do nucleo" { & $pip install -r requirements.txt }
Etapa "Instalando o pacote easse-pt (modo editavel)" { & $pip install -e . }

if ($Spacy) {
    Etapa "Instalando spaCy" { & $pip install spacy }
    Etapa "Baixando o modelo pt_core_news_sm" { & $venvPy -m spacy download pt_core_news_sm }
}

if ($Semantic) {
    Etapa "Instalando a camada semantica (torch, transformers, BERTScore-PT) - demora" {
        & $pip install -r requirements-semantic.txt
    }
}

# --------------------------------------------------------------------------- #
Etapa "Materializando a amostra sintetica" { & $venvPy -m easse_pt.cli corpus-sample }

Etapa "Rodando os testes" {
    $env:PYTHONPATH = "."
    & $venvPy -m unittest discover -s tests
}

Etapa "Diagnostico do ambiente" { & $venvPy -m easse_pt.cli doctor }

# --------------------------------------------------------------------------- #
Write-Host ""
if ($falhas.Count -gt 0) {
    Write-Host "SETUP TERMINOU COM FALHAS em: $($falhas -join ', ')" -ForegroundColor Red
    $code = 1
} else {
    Write-Host "SETUP CONCLUIDO SEM ERROS." -ForegroundColor Green
    $code = 0
}
Write-Host "Log completo: $log"
Write-Host "Para usar o ambiente:  .\.venv\Scripts\Activate.ps1"

Stop-Transcript | Out-Null
exit $code
