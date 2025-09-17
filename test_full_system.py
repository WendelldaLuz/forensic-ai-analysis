#!/usr/bin/env python3
"""
Teste Completo de Funcionalidade do Sistema
"""

import sys
import os
import numpy as np
import tempfile
import traceback

def test_streamlit_compatibility():
    """Testa se arquivos são compatíveis com Streamlit"""
    
    print("🌐 TESTANDO COMPATIBILIDADE STREAMLIT")
    print("=" * 50)
    
    python_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    for py_file in python_files:
        print(f"🔍 Verificando {py_file}...")
        
        try:
            # Verificar sintaxe
            with open(py_file, 'r') as f:
                content = f.read()
            
            # Compilar para verificar sintaxe
            compile(content, py_file, 'exec')
            print(f"✅ Sintaxe OK")
            
            # Verificar se tem funções Streamlit
            if 'import streamlit' in content or 'st.' in content:
                print(f"✅ Compatível com Streamlit")
            else:
                print(f"ℹ️ Não é arquivo Streamlit")
            
        except SyntaxError as e:
            print(f"❌ Erro de sintaxe: {e}")
        except Exception as e:
            print(f"⚠️ Aviso: {e}")

def test_ai_modules():
    """Testa módulos de IA"""
    
    print(f"\n🧠 TESTANDO MÓDULOS DE IA")
    print("=" * 50)
    
    sys.path.append('src')
    
    modules_to_test = [
        ('evolutionary_ai_engine', 'EnhancedForensicAI'),
        ('main', 'main')
    ]
    
    for module_name, class_or_function in modules_to_test:
        try:
            print(f"🔍 Testando {module_name}...")
            
            module = __import__(module_name)
            
            if hasattr(module, class_or_function):
                print(f"✅ {class_or_function} encontrado")
                
                # Testar inicialização se for classe
                if class_or_function != 'main':
                    try:
                        obj = getattr(module, class_or_function)()
                        print(f"✅ {class_or_function} inicializado")
                    except Exception as e:
                        print(f"⚠️ Erro na inicialização: {e}")
            else:
                print(f"❌ {class_or_function} não encontrado")
                
        except ImportError as e:
            print(f"❌ Erro ao importar {module_name}: {e}")
        except Exception as e:
            print(f"❌ Erro inesperado: {e}")

def test_data_processing():
    """Testa processamento de dados"""
    
    print(f"\n📊 TESTANDO PROCESSAMENTO DE DADOS")
    print("=" * 50)
    
    try:
        # Simular dados DICOM
        print("🔍 Criando dados DICOM simulados...")
        test_image = np.random.randint(-1000, 1000, (128, 128), dtype=np.int16)
        print(f"✅ Dados criados: {test_image.shape}, faixa: {test_image.min()} a {test_image.max()}")
        
        # Testar processamento básico
        print("🔍 Testando processamento básico...")
        mean_hu = np.mean(test_image)
        std_hu = np.std(test_image)
        gas_volume = np.sum(test_image < -100) / test_image.size
        
        print(f"✅ Densidade média: {mean_hu:.1f} HU")
        print(f"✅ Desvio padrão: {std_hu:.1f} HU")
        print(f"✅ Volume gasoso: {gas_volume:.2%}")
        
        # Testar análise de gradientes
        print("🔍 Testando análise de gradientes...")
        grad_x = np.gradient(test_image.astype(float), axis=1)
        grad_y = np.gradient(test_image.astype(float), axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        edge_density = np.mean(gradient_mag)
        
        print(f"✅ Densidade de bordas: {edge_density:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        traceback.print_exc()
        return False

def create_missing_files():
    """Cria arquivos que estão faltando"""
    
    print(f"\n🔧 CRIANDO ARQUIVOS FALTANDO")
    print("=" * 50)
    
    # Criar __init__.py se não existir
    init_files = [
        'src/__init__.py',
        'tests/__init__.py'
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            with open(init_file, 'w') as f:
                f.write('"""Pacote do sistema de IA forense"""')
            print(f"✅ Criado: {init_file}")
    
    # Criar diretórios necessários
    directories = ['data', 'logs', 'models', 'config', 'security', 'tests']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Diretório criado: {directory}")
    
    # Criar arquivo de configuração básico se não existir
    if not os.path.exists('data/learning_database.json'):
        basic_config = {
            "analyses_performed": 0,
            "accuracy_history": [],
            "learned_patterns": {},
            "system_version": "1.0.0"
        }
        
        with open('data/learning_database.json', 'w') as f:
            json.dump(basic_config, f, indent=2)
        print("✅ Base de dados de aprendizado criada")

def main():
    """Função principal de verificação"""
    
    print("🔬 FORENSIC AI SYSTEM - VERIFICAÇÃO COMPLETA")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 60)
    
    # Criar arquivos faltando primeiro
    create_missing_files()
    
    # Executar todas as verificações
    missing_req, missing_opt = check_file_structure()
    missing_critical, missing_optional = check_dependencies()
    ai_working = check_ai_functionality()
    streamlit_ok = check_streamlit_compatibility()
    check_data_directories()
    basic_ok = test_data_processing()
    
    # Verificar arquivos vazios
    print(f"\n" + "="*60)
    os.system('python check_empty_files.py')
    
    # Status final
    print(f"\n🎯 STATUS FINAL")
    print("=" * 30)
    
    if (len(missing_req) == 0 and len(missing_critical) == 0 and 
        ai_working and streamlit_ok and basic_ok):
        
        print("🎉 SISTEMA 100% FUNCIONAL!")
        print("🚀 COMANDOS PARA USAR:")
        print("   streamlit run src/main.py")
        print("   streamlit run src/main_integrated.py  # Se disponível")
        
    else:
        print("⚠️ SISTEMA PRECISA DE ATENÇÃO")
        print("🔧 PROBLEMAS ENCONTRADOS:")
        
        if missing_req:
            print(f"   📁 Arquivos faltando: {missing_req}")
        if missing_critical:
            print(f"   📦 Dependências faltando: {missing_critical}")
        if not ai_working:
            print("   🧠 IA não está funcionando")
        if not streamlit_ok:
            print("   🌐 Problema com Streamlit")

if __name__ == "__main__":
    from datetime import datetime
    main()
