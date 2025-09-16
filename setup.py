"""
Setup script for Forensic AI Analysis System
Desenvolvido por: Wendell da Luz Silva
"""

from setuptools import setup, find_packages
import os

# Lê o README
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Sistema Autônomo de IA para Análise Forense"

# Lê os requisitos
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh 
                   if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "streamlit>=1.28.0",
            "pydicom>=2.4.0", 
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scipy>=1.10.0",
            "scikit-image>=0.20.0",
            "matplotlib>=3.7.0",
            "plotly>=5.15.0",
            "Pillow>=10.0.0"
        ]

setup(
    # Informações básicas
    name="forensic-ai-analysis",
    version="1.0.0",
    author="Wendell da Luz Silva",
    author_email="wendell@forensic-ai.com",
    
    # Descrição
    description="🔬 Sistema Autônomo de IA para Análise Forense - Análise Post-Mortem Inteligente",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    
    # URLs do projeto
    url="https://github.com/WendelldaLuz/forensic-ai-analysis",
    project_urls={
        "Bug Reports": "https://github.com/WendelldaLuz/forensic-ai-analysis/issues",
        "Source Code": "https://github.com/WendelldaLuz/forensic-ai-analysis",
        "Documentation": "https://github.com/WendelldaLuz/forensic-ai-analysis/wiki",
        "Changelog": "https://github.com/WendelldaLuz/forensic-ai-analysis/releases",
    },
    
    # Configuração de pacotes
    packages=find_packages(),
    package_dir={"": "."},
    py_modules=["src.main"],
    
    # Classificadores
    classifiers=[
        # Status de desenvolvimento
        "Development Status :: 4 - Beta",
        
        # Público-alvo
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Education",
        
        # Tópicos
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # Licença
        "License :: OSI Approved :: MIT License",
        
        # Versões do Python
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        
        # Sistema operacional
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        
        # Interface
        "Environment :: Web Environment",
        "Environment :: Console",
        
        # Linguagem natural
        "Natural Language :: Portuguese (Brazilian)",
        "Natural Language :: English",
    ],
    
    # Requisitos de Python
    python_requires=">=3.8",
    
    # Dependências
    install_requires=read_requirements(),
    
    # Dependências extras
    extras_require={
        # Funcionalidades completas
        "full": [
            "opencv-python>=4.8.0",
            "reportlab>=4.0.0",
        ],
        
        # Desenvolvimento
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        
        # Testes
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "coverage>=7.2.0",
        ],
        
        # Documentação
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        
        # Deploy
        "deploy": [
            "gunicorn>=21.0.0",
            "docker>=6.1.0",
            "kubernetes>=27.2.0",
        ]
    },
    
    # Scripts de linha de comando
    entry_points={
        "console_scripts": [
            # Comando principal
            "forensic-ai=src.main:main",
            
            # Comandos auxiliares
            "forensic-ai-install=src.utils.installer:main",
            "forensic-ai-test=src.utils.tester:main",
        ],
    },
    
    # Palavras-chave
    keywords=[
        "forensic", "analysis", "ai", "artificial-intelligence",
        "dicom", "medical-imaging", "post-mortem", "interval",
        "streamlit", "python", "healthcare", "legal",
        "autopsy", "pathology", "radiology", "imaging"
    ],
    
    # Arquivos de dados
    include_package_data=True,
    package_data={
        "": [
            "*.md", "*.txt", "*.yml", "*.yaml", "*.json",
            "*.cfg", "*.ini", "*.toml"
        ],
        "src": ["*.py", "**/*.py"],
        "config": ["*.json", "*.yaml", "*.yml"],
        "data": ["samples/*"],
        "docs": ["*.md", "*.rst"],
        "scripts": ["*.sh", "*.bat"],
    },
    
    # Configurações adicionais
    zip_safe=False,
    platforms=["any"],
    
    # Metadados adicionais
    maintainer="Wendell da Luz Silva",
    maintainer_email="wendell@forensic-ai.com",
    
    # Status do projeto
    project_status="4 - Beta",
    
    # Informações de licença
    license="MIT",
    license_files=["LICENSE"],
    
    # Configurações do setuptools
    options={
        "bdist_wheel": {
            "universal": False,
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Informações pós-instalação
def print_post_install_info():
    """Mostra informações após instalação"""
    print("\n" + "="*60)
    print("🔬 FORENSIC AI ANALYSIS SYSTEM")
    print("="*60)
    print("✅ Instalação concluída com sucesso!")
    print("\n📚 Como usar:")
    print("   forensic-ai                    # Executar sistema")
    print("   python -m src.main             # Executar diretamente")
    print("   streamlit run src/main.py      # Executar com Streamlit")
    print("\n🔧 Comandos úteis:")
    print("   forensic-ai-test               # Executar testes")
    print("   forensic-ai-install            # Reinstalar dependências")
    print("\n📖 Documentação:")
    print("   https://github.com/WendelldaLuz/forensic-ai-analysis")
    print("\n🆘 Suporte:")
    print("   Issues: https://github.com/WendelldaLuz/forensic-ai-analysis/issues")
    print("   Email: wendell@forensic-ai.com")
    print("="*60)

# Executar informações pós-instalação se for instalação
if __name__ == "__main__":
    print_post_install_info()
