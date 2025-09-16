#!/bin/bash

# ==============================================================================
# FORENSIC AI ANALYSIS SYSTEM - SCRIPT DE EXECUÇÃO
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

# Emojis
ROCKET="🚀"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
INFO="ℹ️"
MICROSCOPE="🔬"
GLOBE="🌐"
GEAR="⚙️"

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

# Banner de inicialização
print_startup_banner() {
    clear
    echo -e "${CYAN}"
    echo "=============================================================="
    echo "         ${MICROSCOPE} FORENSIC AI ANALYSIS SYSTEM ${MICROSCOPE}"
    echo "=============================================================="
    echo "    Sistema Autônomo de IA para Análise Forense"
    echo "    Desenvolvido por: Wendell da Luz Silva"
    echo "    Status: Iniciando sistema..."
    echo "=============================================================="
    echo -e "${NC}"
}

# Verificar se está no diretório correto
check_project_directory() {
    if [[ ! -f "src/main.py" ]]; then
        log_error "Arquivo src/main.py não encontrado!"
        log_info "Execute este script a partir do diretório raiz do projeto"
        exit 1
    fi
    
    if [[ ! -f "requirements.txt" ]]; then
        log_error "Arquivo requirements.txt não encontrado!"
        log_info "Execute o script de instalação primeiro: ./scripts/install.sh"
        exit 1
    fi
    
    log_success "Diretório do projeto verificado"
}

# Verificar e ativar ambiente virtual
activate_virtual_environment() {
    log_step "Verificando ambiente virtual..."
    
    # Tentar diferentes localizações do ambiente virtual
    if [[ -f "venv/bin/activate" ]]; then
        log_info "Ativando ambiente virtual (Unix)..."
        source venv/bin/activate
        log_success "Ambiente virtual ativado"
        return 0
    elif [[ -f "venv/Scripts/activate" ]]; then
        log_info "Ativando ambiente virtual (Windows)..."
        source venv/Scripts/activate
        log_success "Ambiente virtual ativado"
        return 0
    elif [[ -f ".venv/bin/activate" ]]; then
        log_info "Ativando ambiente virtual (.venv/bin)..."
        source .venv/bin/activate
        log_success "Ambiente virtual ativado"
        return 0
    elif [[ -f ".venv/Scripts/activate" ]]; then
        log_info "Ativando ambiente virtual (.venv/Scripts)..."
        source .venv/Scripts/activate
        log_success "Ambiente virtual ativado"
        return 0
    else
        log_warning "Ambiente virtual não encontrado!"
        log_info "Executando sem ambiente virtual..."
        log_info "Para criar um ambiente virtual, execute: ./scripts/install.sh"
        return 1
    fi
}

# Verificar Python
check_python() {
    log_step "Verificando Python..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
    else
        log_error "Python não encontrado!"
        log_info "Instale Python 3.8+ ou execute: ./scripts/install.sh"
        exit 1
    fi
    
    # Verificar versão
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 8 ]]; then
        log_success "Python $PYTHON_VERSION encontrado"
    else
        log_error "Python 3.8+ é necessário. Versão encontrada: $PYTHON_VERSION"
        exit 1
    fi
}

# Verificar dependências críticas
check_critical_dependencies() {
    log_step "Verificando dependências críticas..."
    
    critical_deps=("streamlit" "pydicom" "numpy" "pandas")
    missing_deps=()
    
    for dep in "${critical_deps[@]}"; do
        if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Dependências faltando: ${missing_deps[*]}"
        log_info "Execute o script de instalação: ./scripts/install.sh"
        
        # Tentar instalação rápida
        log_warning "Tentando instalação rápida das dependências faltando..."
        for dep in "${missing_deps[@]}"; do
            log_info "Instalando $dep..."
            pip install "$dep" --quiet
        done
        
        # Verificar novamente
        remaining_deps=()
        for dep in "${missing_deps[@]}"; do
            if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
                remaining_deps+=("$dep")
            fi
        done
        
        if [[ ${#remaining_deps[@]} -gt 0 ]]; then
            log_error "Ainda faltam dependências: ${remaining_deps[*]}"
            exit 1
        else
            log_success "Dependências instaladas com sucesso!"
        fi
    else
        log_success "Todas as dependências críticas estão disponíveis"
    fi
}

# Verificar dependências opcionais
check_optional_dependencies() {
    log_step "Verificando dependências opcionais..."
    
    optional_deps=("cv2:opencv-python" "reportlab:reportlab")
    
    for dep_info in "${optional_deps[@]}"; do
        import_name=$(echo $dep_info | cut -d':' -f1)
        package_name=$(echo $dep_info | cut -d':' -f2)
        
        if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
            log_success "$package_name disponível"
        else
            log_warning "$package_name não disponível (funcionalidade limitada)"
        fi
    done
}

# Verificar porta disponível
check_port_availability() {
    local port=${1:-8501}
    
    if command -v netstat &> /dev/null; then
        if netstat -an | grep -q ":$port.*LISTEN"; then
            log_warning "Porta $port já está em uso"
            return 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -an | grep -q ":$port.*LISTEN"; then
            log_warning "Porta $port já está em uso"
            return 1
        fi
    elif command -v lsof &> /dev/null; then
        if lsof -i :$port &> /dev/null; then
            log_warning "Porta $port já está em uso"
            return 1
        fi
    fi
    
    return 0
}

# Encontrar porta disponível
find_available_port() {
    local start_port=${1:-8501}
    local max_attempts=10
    
    for ((i=0; i<max_attempts; i++)); do
        local port=$((start_port + i))
        if check_port_availability $port; then
            echo $port
            return 0
        fi
    done
    
    # Se não encontrar porta disponível, usar a padrão mesmo assim
    echo $start_port
    return 1
}

# Preparar ambiente de execução
prepare_environment() {
    log_step "Preparando ambiente de execução..."
    
    # Criar diretórios necessários se não existirem
    local dirs=("logs" "temp" "data/samples")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Diretório criado: $dir"
        fi
    done
    
    # Limpar arquivos temporários antigos
    if [[ -d "temp" ]]; then
        find temp -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
        find temp -name "*.dcm" -mtime +1 -delete 2>/dev/null || true
    fi
    
    log_success "Ambiente preparado"
}

# Verificar atualizações (opcional)
check_for_updates() {
    if command -v git &> /dev/null && [[ -d ".git" ]]; then
        log_step "Verificando atualizações..."
        
        # Fetch remoto silenciosamente
        git fetch origin main --quiet 2>/dev/null || true
        
        # Comparar commits
        LOCAL=$(git rev-parse HEAD 2>/dev/null || echo "")
        REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "")
        
        if [[ "$LOCAL" != "$REMOTE" && -n "$REMOTE" ]]; then
            log_warning "Nova versão disponível!"
            log_info "Execute 'git pull origin main' para atualizar"
        fi
    fi
}

# Iniciar aplicação Streamlit
start_streamlit_app() {
    local port=${1:-8501}
    local host=${2:-localhost}
    
    log_step "Iniciando aplicação Streamlit..."
    
    # Configurações do Streamlit
    export STREAMLIT_SERVER_PORT=$port
    export STREAMLIT_SERVER_ADDRESS=$host
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    export STREAMLIT_SERVER_ENABLE_CORS=false
    export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
    
    # Argumentos do Streamlit
    streamlit_args=(
        "run"
        "src/main.py"
        "--server.port=$port"
        "--server.address=$host"
        "--server.headless=false"
        "--browser.gatherUsageStats=false"
        "--server.enableCORS=false"
        "--server.enableXsrfProtection=false"
        "--theme.base=dark"
        "--theme.primaryColor=#667eea"
        "--theme.backgroundColor=#1e1e1e"
        "--theme.secondaryBackgroundColor=#262730"
        "--theme.textColor=#ffffff"
    )
    
    log_success "Configuração concluída!"
    echo ""
    echo -e "${GREEN}=============================================================="
    echo -e "           ${ROCKET} SISTEMA INICIADO COM SUCESSO! ${ROCKET}"
    echo -e "==============================================================${NC}"
    echo ""
    echo -e "${WHITE}Acesse a aplicação em:${NC}"
    echo -e "  ${CYAN}${GLOBE} http://$host:$port${NC}"
    echo ""
    echo -e "${WHITE}Como usar:${NC}"
    echo -e "  1. ${BLUE}Carregue um arquivo DICOM${NC}"
    echo -e "  2. ${BLUE}Configure parâmetros ambientais${NC}" 
    echo -e "  3. ${BLUE}Clique em 'Analisar com IA'${NC}"
    echo -e "  4. ${BLUE}Visualize os resultados automáticos${NC}"
    echo ""
    echo -e "${YELLOW}Para parar o sistema: ${WHITE}Ctrl+C${NC}"
    echo ""
    echo -e "${PURPLE}Aguarde o carregamento da aplicação...${NC}"
    echo ""
    
    # Iniciar Streamlit
    $PYTHON_CMD -m streamlit "${streamlit_args[@]}"
}

# Função para limpeza ao sair
cleanup_on_exit() {
    echo ""
    log_info "Finalizando sistema..."
    
    # Limpar arquivos temporários
    if [[ -d "temp" ]]; then
        find temp -name "*.tmp" -delete 2>/dev/null || true
    fi
    
    log_success "Sistema finalizado com sucesso!"
    exit 0
}

# Função principal
main() {
    # Configurar trap para limpeza ao sair
    trap cleanup_on_exit SIGINT SIGTERM EXIT
    
    # Banner inicial
    print_startup_banner
    
    # Verificações pré-execução
    check_project_directory
    check_python
    
    # Ativar ambiente virtual (opcional)
    activate_virtual_environment
    
    # Verificar dependências
    check_critical_dependencies
    check_optional_dependencies
    
    # Preparar ambiente
    prepare_environment
    
    # Verificar atualizações (opcional)
    check_for_updates
    
    # Encontrar porta disponível
    log_step "Verificando disponibilidade de porta..."
    PORT=$(find_available_port 8501)
    
    if [[ $PORT != "8501" ]]; then
        log_warning "Porta 8501 em uso, usando porta $PORT"
    else
        log_success "Porta $PORT disponível"
    fi
    
    # Iniciar aplicação
    start_streamlit_app $PORT "localhost"
}

# Verificar argumentos da linha de comando
case "${1:-}" in
    --help|-h)
        echo "Uso: $0 [opções]"
        echo ""
        echo "Opções:"
        echo "  --help, -h     Mostrar esta ajuda"
        echo "  --version, -v  Mostrar versão"
        echo "  --port PORT    Especificar porta (padrão: 8501)"
        echo "  --host HOST    Especificar host (padrão: localhost)"
        echo ""
        exit 0
        ;;
    --version|-v)
        echo "Forensic AI Analysis System v1.0.0"
        echo "Desenvolvido por: Wendell da Luz Silva"
        exit 0
        ;;
    --port)
        if [[ -n "${2:-}" ]]; then
            CUSTOM_PORT="$2"
            shift 2
        else
            log_error "Porta não especificada após --port"
            exit 1
        fi
        ;;
    --host)
        if [[ -n "${2:-}" ]]; then
            CUSTOM_HOST="$2"
            shift 2
        else
            log_error "Host não especificado após --host"
            exit 1
        fi
        ;;
    "")
        # Sem argumentos, continuar normalmente
        ;;
    *)
        log_error "Argumento desconhecido: $1"
        log_info "Use --help para ver opções disponíveis"
        exit 1
        ;;
esac

# Executar função principal
main
