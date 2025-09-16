@echo off
setlocal enabledelayedexpansion

REM ==============================================================================
REM FORENSIC AI ANALYSIS SYSTEM - SCRIPT DE EXECUCAO WINDOWS
REM Desenvolvido por: Wendell da Luz Silva
REM Sistema Autonomo de IA para Analise Forense
REM ==============================================================================

title Forensic AI Analysis System - Sistema de IA Forense

REM Banner inicial
echo.
echo ==============================================================
echo          🔬 FORENSIC AI ANALYSIS SYSTEM 🔬
echo ==============================================================
echo     Sistema Autonomo de IA para Analise Forense
echo     Desenvolvido por: Wendell da Luz Silva
echo     Status: Iniciando sistema...
echo ==============================================================
echo.

REM Verificar se esta no diretorio correto
if not exist "src\main.py" (
    echo ❌ Arquivo src\main.py nao encontrado!
    echo ℹ️ Execute este script a partir do diretorio raiz do projeto
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ Arquivo requirements.txt nao encontrado!
    echo ℹ️ Execute o script de instalacao primeiro
    pause
    exit /b 1
)

echo ✅ Diretorio do projeto verificado

REM Verificar e ativar ambiente virtual
echo ⚙️ Verificando ambiente virtual...

if exist "venv\Scripts\activate.bat" (
    echo ℹ️ Ativando ambiente virtual...
    call venv\Scripts\activate.bat
    echo ✅ Ambiente virtual ativado
) else if exist ".venv\Scripts\activate.bat" (
    echo ℹ️ Ativando ambiente virtual (.venv)...
    call .venv\Scripts\activate.bat
    echo ✅ Ambiente virtual ativado
) else (
    echo ⚠️ Ambiente virtual nao encontrado!
    echo ℹ️ Executando sem ambiente virtual...
    echo ℹ️ Para criar um ambiente virtual, execute o script de instalacao
)

REM Verificar Python
echo ⚙️ Verificando Python...

python --version >nul 2>&1
if !errorlevel! == 0 (
    set PYTHON_CMD=python
    echo ✅ Python encontrado
) else (
    python3 --version >nul 2>&1
    if !errorlevel! == 0 (
        set PYTHON_CMD=python3
        echo ✅ Python3 encontrado
    ) else (
        echo ❌ Python nao encontrado!
        echo ℹ️ Instale Python 3.8+ ou execute o script de instalacao
        pause
        exit /b 1
    )
)

REM Verificar dependencias criticas
echo ⚙️ Verificando dependencias criticas...

%PYTHON_CMD% -c "import streamlit" 2>nul
if !errorlevel! neq 0 (
    echo ❌ Streamlit nao encontrado!
    echo ℹ️ Executando instalacao rapida...
    pip install streamlit --quiet
)

%PYTHON_CMD% -c "import pydicom" 2>nul
if !errorlevel! neq 0 (
    echo ❌ PyDICOM nao encontrado!
    echo ℹ️ Executando instalacao rapida...
    pip install pydicom --quiet
)

echo ✅ Dependencias verificadas

REM Preparar ambiente
echo ⚙️ Preparando ambiente de execucao...

if not exist "logs" mkdir logs
if not exist "temp" mkdir temp
if not exist "data\samples" mkdir data\samples

echo ✅ Ambiente preparado

REM Configurar variaveis do Streamlit
set STREAMLIT_SERVER_PORT=8501
set STREAMLIT_SERVER_ADDRESS=localhost
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM Mostrar informacoes de inicio
echo.
echo ==============================================================
echo            🚀 SISTEMA INICIADO COM SUCESSO! 🚀
echo ==============================================================
echo.
echo Acesse a aplicacao em:
echo   🌐 http://localhost:8501
echo.
echo Como usar:
echo   1. 📁 Carregue um arquivo DICOM
echo   2. 🌡️ Configure parametros ambientais  
echo   3. 🚀 Clique em 'Analisar com IA'
echo   4. 📊 Visualize os resultados automaticos
echo.
echo Para parar o sistema: Ctrl+C
echo.
echo Aguarde o carregamento da aplicacao...
echo.

REM Iniciar Streamlit
%PYTHON_CMD% -m streamlit run src\main.py ^
    --server.port=8501 ^
    --server.address=localhost ^
    --server.headless=false ^
    --browser.gatherUsageStats=false ^
    --server.enableCORS=false ^
    --theme.base=dark ^
    --theme.primaryColor=#667eea ^
    --theme.backgroundColor=#1e1e1e ^
    --theme.secondaryBackgroundColor=#262730 ^
    --theme.textColor=#ffffff

REM Limpeza ao sair
echo.
echo ℹ️ Finalizando sistema...
echo ✅ Sistema finalizado com sucesso!
pause
