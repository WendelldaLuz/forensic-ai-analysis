#!/usr/bin/env python3
"""
Script de Verificação Completa do Sistema
Verifica arquivos, dependências e funcionalidades
"""

import os
import sys
import json
import importlib
from pathlib import Path
import subprocess

def check_file_structure():
    """Verifica estrutura de arquivos"""
    
    print("🔍 VERIFICANDO ESTRUTURA DE ARQUIVOS")
    print("=" * 50)
    
    # Arquivos obrigatórios
    required_files = {
        'src/main.py': 'Arquivo principal',
        'src/evolutionary_ai_engine.py': 'IA Evolutiva',
        'src/__init__.py': 'Init do src',
        'requirements.txt': 'Dependências',
        'setup.py': 'Configuração Python',
        'README.md': 'Documentação'
    }
    
    # Arquivos opcionais mas importantes
    optional_files = {
        'src/main_integrated.py': 'Sistema integrado',
        'src/security_system.py': 'Sistema de segurança', 
        'src/professional_ui.py': 'Interface profissional',
        'src/advanced_report_generator.py': 'Gerador de relatórios',
        'config/ai_learning_config.json': 'Config de aprendizado',
        'data/learning_database.json': 'Base de dados IA',
        'tests/test_ai_engine.py': 'Testes da IA',
        'tests/__init__.py': 'Init dos testes'
    }
    
    # Verificar arquivos obrigatórios
    missing_required = []
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size} bytes) - {description}")
        else:
            print(f"❌ {file_path} - {description} [FALTANDO]")
            missing_required.append(file_path)
    
    print(f"\n📋 ARQUIVOS OPCIONAIS:")
    missing_optional = []
    for file_path, description in optional_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            if size < 100:
                print(f"⚠️ {file_path} ({size} bytes) - {description} [MUITO PEQUENO]")
            else:
                print(f"✅ {file_path} ({size} bytes) - {description}")
        else:
            print(f"❌ {file_path} - {description} [FALTANDO]")
            missing_optional.append(file_path)
    
    return missing_required, missing_optional

def check_dependencies():
    """Verifica dependências Python"""
    
    print(f"\n🐍 VERIFICANDO DEPENDÊNCIAS PYTHON")
    print("=" * 50)
    
    critical_deps = [
        'streamlit', 'pydicom', 'numpy', 'pandas', 'matplotlib', 
        'plotly', 'scipy', 'PIL'
    ]
    
    optional_deps = [
        'cv2', 'reportlab', 'sklearn', 'skimage'
    ]
    
    missing_critical = []
    missing_optional = []
    
    # Verificar dependências críticas
    for dep in critical_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'N/A')
            print(f"✅ {dep} v{version}")
        except ImportError:
            print(f"❌ {dep} [FALTANDO - CRÍTICO]")
            missing_critical.append(dep)
    
    # Verificar dependências opcionais
    print(f"\n📦 DEPENDÊNCIAS OPCIONAIS:")
    for dep in optional_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'N/A')
            print(f"✅ {dep} v{version}")
        except ImportError:
            print(f"⚠️ {dep} [FALTANDO - OPCIONAL]")
            missing_optional.append(dep)
    
    return missing_critical, missing_optional

def check_ai_functionality():
    """Verifica funcionalidade da IA"""
    
    print(f"\n🧠 VERIFICANDO FUNCIONALIDADE DA IA")
    print("=" * 50)
    
    try:
        sys.path.append('src')
        from evolutionary_ai_engine import EnhancedForensicAI, RAIndexCalculator
        
        print("✅ Importação da IA bem-sucedida")
        
        # Testar RA-Index
        ra_calc = RAIndexCalculator()
        test_classification = {
            "Cavidades Cardíacas": "II",
            "Parênquima Hepático e Vasos": "I",
            "Veia Inominada Esquerda": "0",
            "Aorta Abdominal": "0", 
            "Parênquima Renal": "I",
            "Vértebra L3": "0",
            "Tecidos Subcutâneos Peitorais": "0"
        }
        
        ra_result = ra_calc.calculate_ra_index(test_classification)
        
        if 'ra_index' in ra_result:
            print(f"✅ RA-Index calculado: {ra_result['ra_index']}")
        else:
            print("❌ Falha no cálculo do RA-Index")
            return False
        
        # Testar IA completa
        ai_system = EnhancedForensicAI()
        print("✅ Sistema de IA inicializado")
        
        # Testar análise com dados simulados
        test_image = np.random.randint(-500, 500, (50, 50), dtype=np.int16)
        env_params = {'temperature': 25, 'humidity': 60}
        
        analysis_result = ai_system.comprehensive_analysis(test_image, env_params)
        
        if 'ra_index' in analysis_result:
            print("✅ Análise completa funcionando")
            return True
        else:
            print("❌ Falha na análise completa")
            return False
            
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro na IA: {e}")
        return False

def check_streamlit_compatibility():
    """Verifica compatibilidade com Streamlit"""
    
    print(f"\n🌐 VERIFICANDO STREAMLIT")
    print("=" * 50)
    
    try:
        import streamlit as st
        print(f"✅ Streamlit v{st.__version__} disponível")
        
        # Verificar se main.py é válido para Streamlit
        if os.path.exists('src/main.py'):
            with open('src/main.py', 'r') as f:
                content = f.read()
                
            if 'st.' in content and 'def main():' in content:
                print("✅ main.py compatível com Streamlit")
            else:
                print("⚠️ main.py pode ter problemas de compatibilidade")
        
        return True
        
    except ImportError:
        print("❌ Streamlit não disponível")
        return False

def check_data_directories():
    """Verifica diretórios de dados"""
    
    print(f"\n📁 VERIFICANDO DIRETÓRIOS DE DADOS")
    print("=" * 50)
    
    required_dirs = ['data', 'logs', 'models', 'config', 'security']
    
    for directory in required_dirs:
        if os.path.exists(directory):
            files_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
            print(f"✅ {directory}/ ({files_count} arquivos)")
        else:
            print(f"❌ {directory}/ [FALTANDO]")
            os.makedirs(directory, exist_ok=True)
            print(f"🔧 {directory}/ criado")

def test_basic_functionality():
    """Testa funcionalidades básicas"""
    
    print(f"\n⚡ TESTANDO FUNCIONALIDADES BÁSICAS")
    print("=" * 50)
    
    try:
        # Teste 1: Importação numpy
        import numpy as np
        test_array = np.random.rand(10, 10)
        print(f"✅ NumPy funcionando - Array {test_array.shape}")
        
        # Teste 2: Importação pandas
        import pandas as pd
        test_df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        print(f"✅ Pandas funcionando - DataFrame {test_df.shape}")
        
        # Teste 3: Importação plotly
        import plotly.graph_objects as go
        fig = go.Figure()
        print("✅ Plotly funcionando")
        
        # Teste 4: Importação scipy
        from scipy import stats
        test_stat = stats.norm.rvs(size=10)
        print(f"✅ SciPy funcionando - Stats OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades básicas: {e}")
        return False

def generate_system_report():
    """Gera relatório completo do sistema"""
    
    print(f"\n📊 GERANDO RELATÓRIO DO SISTEMA")
    print("=" * 50)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'platform': sys.platform,
        'file_check': {},
        'dependency_check': {},
        'functionality_check': {}
    }
    
    # Executar verificações
    missing_req, missing_opt = check_file_structure()
    missing_critical, missing_optional = check_dependencies()
    ai_working = check_ai_functionality()
    streamlit_ok = check_streamlit_compatibility()
    basic_ok = test_basic_functionality()
    
    # Compilar relatório
    report['file_check'] = {
        'missing_required': missing_req,
        'missing_optional': missing_opt
    }
    
    report['dependency_check'] = {
        'missing_critical': missing_critical,
        'missing_optional': missing_optional
    }
    
    report['functionality_check'] = {
        'ai_working': ai_working,
        'streamlit_compatible': streamlit_ok,
        'basic_functions': basic_ok
    }
    
    # Status geral
    overall_status = (
        len(missing_req) == 0 and 
        len(missing_critical) == 0 and
        ai_working and 
        streamlit_ok and 
        basic_ok
    )
    
    report['overall_status'] = 'READY' if overall_status else 'NEEDS_ATTENTION'
    
    # Salvar relatório
    with open('system_check_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 RELATÓRIO SALVO: system_check_report.json")
    
    # Mostrar status final
    print(f"\n🎯 STATUS FINAL DO SISTEMA")
    print("=" * 50)
    
    if overall_status:
        print("🎉 SISTEMA PRONTO PARA USO!")
        print("🚀 Execute: streamlit run src/main.py")
    else:
        print("⚠️ SISTEMA PRECISA DE ATENÇÃO")
        if missing_req:
            print(f"❌ Arquivos obrigatórios faltando: {missing_req}")
        if missing_critical:
            print(f"❌ Dependências críticas faltando: {missing_critical}")
        if not ai_working:
            print("❌ IA não está funcionando")
        if not streamlit_ok:
            print("❌ Problema com Streamlit")
    
    return overall_status

if __name__ == "__main__":
    print("🔬 FORENSIC AI SYSTEM - VERIFICAÇÃO COMPLETA")
    print("=" * 60)
    
    # Verificar diretórios primeiro
    check_data_directories()
    
    # Executar verificação completa
    system_ready = generate_system_report()
    
    print(f"\n🏁 VERIFICAÇÃO CONCLUÍDA")
    
    if system_ready:
        print("🎊 Sistema 100% funcional!")
    else:
        print("🔧 Sistema precisa de ajustes")
