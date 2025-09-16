"""
Forensic AI Analysis System
Sistema Autônomo de IA para Análise Forense

Este módulo contém as funcionalidades principais do sistema de análise forense
com inteligência artificial para estimativa de intervalo post-mortem.

Author: Wendell da Luz Silva
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Wendell da Luz Silva"
__email__ = "wendell@forensic-ai.com"
__description__ = "Sistema Autônomo de IA para Análise Forense"

# Metadados do pacote
__title__ = "forensic-ai-analysis"
__summary__ = "Sistema de IA para análise forense automatizada de imagens DICOM"
__uri__ = "https://github.com/WendelldaLuz/forensic-ai-analysis"
__license__ = "MIT"
__copyright__ = "2024 Wendell da Luz Silva"

# Configurações do sistema
SYSTEM_CONFIG = {
    "name": "Forensic AI Analysis System",
    "version": __version__,
    "author": __author__,
    "description": __description__,
    "supported_formats": ["dcm", "dicom"],
    "min_python_version": "3.8",
    "default_port": 8501,
    "confidence_threshold": 0.8
}

# Verificação de dependências críticas
def check_dependencies():
    """Verifica se as dependências críticas estão disponíveis"""
    
    missing_deps = []
    
    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")
    
    try:
        import pydicom
    except ImportError:
        missing_deps.append("pydicom")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        raise ImportError(f"Dependências faltando: {', '.join(missing_deps)}")
    
    return True

# Função de inicialização
def initialize_system():
    """Inicializa o sistema e verifica configurações"""
    
    print(f"🔬 Inicializando {SYSTEM_CONFIG['name']} v{SYSTEM_CONFIG['version']}")
    print(f"👨‍💻 Desenvolvido por: {SYSTEM_CONFIG['author']}")
    
    # Verificar dependências
    try:
        check_dependencies()
        print("✅ Todas as dependências críticas estão disponíveis")
        return True
    except ImportError as e:
        print(f"❌ Erro de dependências: {e}")
        print("💡 Execute: pip install -r requirements.txt")
        return False

# Exportações principais
__all__ = [
    "__version__",
    "__author__", 
    "__description__",
    "SYSTEM_CONFIG",
    "check_dependencies",
    "initialize_system"
]

# Verificação automática ao importar
if __name__ != "__main__":
    # Verificação silenciosa ao importar
    try:
        check_dependencies()
    except ImportError:
        import warnings
        warnings.warn(
            "Algumas dependências não estão disponíveis. "
            "Execute: pip install -r requirements.txt",
            ImportWarning
        )
