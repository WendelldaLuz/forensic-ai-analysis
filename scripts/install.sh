#!/bin/bash

# ==============================================================================
# FORENSIC AI ANALYSIS SYSTEM - SCRIPT DE INSTALAÇÃO
# Desenvolvido por: Wendell da Luz Silva
# Sistema Autônomo de IA para Análise Forense
# ==============================================================================

set -e  # Para na primeira falha

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Emojis para melhor visualização
ROCKET="🚀"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
INFO="ℹ️"
GEAR="⚙️"
MICROSCOPE="🔬"
PACKAGE="📦"
DOWNLOAD="⬇️"
INSTALL="🔧"

# Banner de boas-vindas
print_banner() {
    echo -e "${CYAN}"
    echo "=============================================================="
    echo "         ${MICROSCOPE} FORENSIC AI ANALYSIS SYSTEM ${MICROSCOPE}"
    echo "=============================================================="
    echo "    Sistema Autônomo de IA para Análise Forense"
    echo "    Desenvolvido por: Wendell da Luz Silva"
    echo "    Versão: 1.0.0"
    echo "=============================================================="
    echo -e "${NC}"
}

# Função para logs coloridos
log_info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

log_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

log_error() {
    echo -e "${RED}${CROSS} $1${NC}"
}

log_step() {
    echo -e "${PURPLE}${GEAR} $1${NC}"
}

# Verificar sistema operacional
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        DISTRO=$(lsb_release -si 2>/dev/null || echo "Unknown")
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macOS"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
        DISTRO="Windows"
    else
        OS="unknown"
        DISTRO="Unknown"
    fi
    
    log_info "Sistema detectado: $DISTRO ($OS)"
}

# Verificar se Python está instalado
check_python() {
    log_step "Verificando instalação do Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            log_success "Python $PYTHON_VERSION encontrado"
            PYTHON_CMD="python3"
            return 0
        else
            log_error "Python 3.8+ é necessário. Versão encontrada: $PYTHON_VERSION"
            return 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
        
        if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
            log_success "Python $PYTHON_VERSION encontrado"
            PYTHON_CMD="python"
            return 0
        else
            log_error "Python 3.8+ é necessário. Versão encontrada: $PYTHON_VERSION"
            return 1
        fi
    else
        log_error "Python não encontrado!"
        return 1
    fi
}

# Instalar Python se necessário
install_python() {
    log_step "Tentando instalar Python..."
    
    case $OS in
        "linux")
            if command -v apt-get &> /dev/null; then
                log_info "Instalando Python via apt-get..."
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip python3-venv
            elif command -v yum &> /dev/null; then
                log_info "Instalando Python via yum..."
                sudo yum install -y python3 python3-pip
            elif command -v dnf &> /dev/null; then
                log_info "Instalando Python via dnf..."
                sudo dnf install -y python3 python3-pip
            elif command -v pacman &> /dev/null; then
                log_info "Instalando Python via pacman..."
                sudo pacman -S python python-pip
            else
                log_error "Gerenciador de pacotes não suportado. Instale Python 3.8+ manualmente."
                return 1
            fi
            ;;
        "macos")
            if command -v brew &> /dev/null; then
                log_info "Instalando Python via Homebrew..."
                brew install python@3.11
            else
                log_warning "Homebrew não encontrado. Visite: https://python.org/downloads/"
                return 1
            fi
            ;;
        *)
            log_error "Sistema não suportado para instalação automática."
            log_info "Por favor, instale Python 3.8+ manualmente de: https://python.org/downloads/"
            return 1
            ;;
    esac
}

# Criar ambiente virtual
create_venv() {
    log_step "Criando ambiente virtual..."
    
    if [[ -d "venv" ]]; then
        log_warning "Ambiente virtual já existe. Removendo..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    
    if [[ $? -eq 0 ]]; then
        log_success "Ambiente virtual criado com sucesso"
    else
        log_error "Falha ao criar ambiente virtual"
        return 1
    fi
}

# Ativar ambiente virtual
activate_venv() {
    log_step "Ativando ambiente virtual..."
    
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        log_success "Ambiente virtual ativado (Unix)"
    elif [[ -f "venv/Scripts/activate" ]]; then
        source venv/Scripts/activate
        log_success "Ambiente virtual ativado (Windows)"
    else
        log_error "Não foi possível encontrar script de ativação do ambiente virtual"
        return 1
    fi
}

# Atualizar pip
upgrade_pip() {
    log_step "Atualizando pip..."
    
    pip install --upgrade pip
    
    if [[ $? -eq 0 ]]; then
        log_success "pip atualizado com sucesso"
    else
        log_warning "Falha ao atualizar pip, mas continuando..."
    fi
}

# Instalar dependências
install_dependencies() {
    log_step "Instalando dependências Python..."
    
    if [[ ! -f "requirements.txt" ]]; then
        log_error "Arquivo requirements.txt não encontrado!"
        return 1
    fi
    
    # Instalar dependências básicas primeiro
    log_info "Instalando dependências básicas..."
    pip install wheel setuptools
    
    # Instalar dependências do projeto
    log_info "Instalando dependências do projeto..."
    pip install -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        log_success "Dependências instaladas com sucesso"
    else
        log_error "Falha ao instalar dependências"
        
        # Tentar instalar dependências críticas individualmente
        log_warning "Tentando instalar dependências críticas individualmente..."
        
        critical_deps=("streamlit" "pydicom" "numpy" "pandas" "scipy" "matplotlib" "plotly")
        
        for dep in "${critical_deps[@]}"; do
            log_info "Instalando $dep..."
            pip install "$dep"
        done
        
        log_warning "Algumas dependências podem ter falhado. Verifique os logs acima."
    fi
}

# Instalar dependências opcionais
install_optional_dependencies() {
    log_step "Instalando dependências opcionais..."
    
    optional_deps=("opencv-python" "reportlab")
    
    for dep in "${optional_deps[@]}"; do
        log_info "Tentando instalar $dep..."
        pip install "$dep" 2>/dev/null
        if [[ $? -eq 0 ]]; then
            log_success "$dep instalado"
        else
            log_warning "$dep não pôde ser instalado (opcional)"
        fi
    done
}

# Criar diretórios necessários
create_directories() {
    log_step "Criando estrutura de diretórios..."
    
    directories=(
        "data/samples"
        "logs"
        "config"
        "temp"
        "reports"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_success "Diretório criado: $dir"
        else
            log_info "Diretório já existe: $dir"
        fi
    done
}

# Configurar permissões
setup_permissions() {
    log_step "Configurando permissões..."
    
    # Tornar scripts executáveis
    if [[ -f "scripts/run.sh" ]]; then
        chmod +x scripts/run.sh
        log_success "Permissões configuradas para scripts/run.sh"
    fi
    
    if [[ -f "scripts/install.sh" ]]; then
        chmod +x scripts/install.sh
        log_success "Permissões configuradas para scripts/install.sh"
    fi
    
    # Configurar permissões de diretórios
    chmod 755 data logs config temp reports 2>/dev/null || true
}

# Testar instalação
test_installation() {
    log_step "Testando instalação..."
    
    # Teste básico de importações
    $PYTHON_CMD -c "
import sys
try:
    import streamlit
    import pydicom  
    import numpy
    import pandas
    import scipy
    import matplotlib
    import plotly
    print('${CHECK} Todas as dependências críticas foram importadas com sucesso!')
except ImportError as e:
    print('${CROSS} Erro ao importar dependência:', str(e))
    sys.exit(1)
" 2>/dev/null

    if [[ $? -eq 0 ]]; then
        log_success "Teste de instalação passou!"
    else
        log_error "Teste de instalação falhou"
        return 1
    fi
}

# Criar arquivo de configuração
create_config() {
    log_step "Criando arquivo de configuração..."
    
    if [[ ! -f "config/settings.json" ]]; then
        cat > config/settings.json << EOF
{
  "app": {
    "name": "Forensic AI Analysis System",
    "version": "1.0.0",
    "description": "Sistema Autônomo de IA para Análise Forense",
    "author": "Wendell da Luz Silva",
    "debug": false
  },
  "analysis": {
    "confidence_threshold": 0.8,
    "max_processing_time": 300,
    "supported_formats": ["dcm", "dicom"],
    "default_temperature": 25,
    "default_humidity": 60
  },
  "ui": {
    "theme": "gradient",
    "sidebar_expanded": true,
    "show_advanced_options": true
  },
  "security": {
    "local_processing_only": true,
    "auto_cleanup": true,
    "log_actions": true
  },
  "paths": {
    "data_dir": "data",
    "logs_dir": "logs", 
    "temp_dir": "temp",
    "reports_dir": "reports"
  }
}
EOF
        log_success "Arquivo de configuração criado"
    else
        log_info "Arquivo de configuração já existe"
    fi
}

# Função principal de instalação
main_install() {
    print_banner
    
    log_info "Iniciando processo de instalação..."
    
    # Detectar sistema operacional
    detect_os
    
    # Verificar/instalar Python
    if ! check_python; then
        log_warning "Python não encontrado ou versão inadequada"
        if ! install_python; then
            log_error "Falha ao instalar Python. Instalação abortada."
            exit 1
        fi
        # Verificar novamente após instalação
        if ! check_python; then
            log_error "Python ainda não está disponível após instalação"
            exit 1
        fi
    fi
    
    # Criar ambiente virtual
    if ! create_venv; then
        log_error "Falha ao criar ambiente virtual"
        exit 1
    fi
    
    # Ativar ambiente virtual  
    if ! activate_venv; then
        log_error "Falha ao ativar ambiente virtual"
        exit 1
    fi
    
    # Atualizar pip
    upgrade_pip
    
    # Instalar dependências
    if ! install_dependencies; then
        log_error "Falha crítica ao instalar dependências"
        exit 1
    fi
    
    # Instalar dependências opcionais
    install_optional_dependencies
    
    # Criar diretórios
    create_directories
    
    # Configurar permissões
    setup_permissions
    
    # Criar configuração
    create_config
    
    # Testar instalação
    if ! test_installation; then
        log_error "Testes de instalação falharam"
        exit 1
    fi
    
    # Sucesso!
    echo -e "${GREEN}"
    echo "=============================================================="
    echo "           ${CHECK} INSTALAÇÃO CONCLUÍDA COM SUCESSO! ${CHECK}"
    echo "=============================================================="
    echo -e "${NC}"
    
    echo -e "${WHITE}Como usar:${NC}"
    echo -e "  ${CYAN}./scripts/run.sh${NC}              # Executar sistema (Unix/Linux/macOS)"
    echo -e "  ${CYAN}scripts/run.bat${NC}               # Executar sistema (Windows)"
    echo -e "  ${CYAN}python -m streamlit run src/main.py${NC}  # Executar diretamente"
    echo ""
    echo -e "${WHITE}Acesse:${NC} ${BLUE}http://localhost:8501${NC}"
    echo ""
    echo -e "${WHITE}Documentação:${NC} ${BLUE}https://github.com/WendelldaLuz/forensic-ai-analysis${NC}"
    echo ""
    echo -e "${YELLOW}${INFO} Certifique-se de ativar o ambiente virtual antes de usar:${NC}"
    echo -e "  ${CYAN}source venv/bin/activate${NC}      # Unix/Linux/macOS" 
    echo -e "  ${CYAN}venv\\Scripts\\activate${NC}        # Windows"
    echo ""
}

# Função de limpeza em caso de erro
cleanup_on_error() {
    log_warning "Limpando arquivos temporários devido a erro..."
    
    # Desativar ambiente virtual se estiver ativo
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate 2>/dev/null || true
    fi
    
    # Remover ambiente virtual parcialmente criado
    if [[ -d "venv" ]]; then
        log_info "Removendo ambiente virtual incompleto..."
        rm -rf venv
    fi
}

# Configurar trap para limpeza em caso de erro
trap cleanup_on_error ERR

# Verificar se está no diretório correto
if [[ ! -f "src/main.py" ]] || [[ ! -f "requirements.txt" ]]; then
    log_error "Execute este script a partir do diretório raiz do projeto"
    log_info "Certifique-se de que os arquivos src/main.py e requirements.txt existem"
    exit 1
fi

# Executar instalação
main_install

# Desabilitar trap de limpeza após sucesso
trap - ERR

exit 0
