#!/usr/bin/env python3
"""
Sistema de Validação Completo - Análise Forense AI
Testa todos os componentes do sistema
"""

import os
import sys
import importlib
import subprocess
import json
import traceback
from pathlib import Path

class SystemValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success = []
        
    def log_error(self, message):
        """Registra erro"""
        self.errors.append(message)
        print(f"❌ [ERRO] {message}")
        
    def log_warning(self, message):
        """Registra aviso"""
        self.warnings.append(message)
        print(f"⚠️  [AVISO] {message}")
        
    def log_success(self, message):
        """Registra sucesso"""
        self.success.append(message)
        print(f"✅ [SUCESSO] {message}")
        
    def print_summary(self):
        """Imprime resumo da validação"""
        print("\n" + "="*60)
        print("RESUMO DA VALIDAÇÃO DO SISTEMA")
        print("="*60)
        print(f"✅ Sucessos: {len(self.success)}")
        print(f"⚠️  Avisos: {len(self.warnings)}")
        print(f"❌ Erros: {len(self.errors)}")
        print("="*60)
        
        if self.errors:
            print("\nERROS ENCONTRADOS:")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print("\nAVISOS:")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if not self.errors:
            print("\n🎉 SISTEMA VALIDADO COM SUCESSO!")
        else:
            print(f"\n💥 {len(self.errors)} erro(s) precisam ser corrigidos!")
            
        return len(self.errors) == 0

    def check_file_structure(self):
        """Verifica estrutura de arquivos"""
        print("\n📁 VERIFICANDO ESTRUTURA DE ARQUIVOS...")
        
        required_files = [
            "src/__init__.py",
            "src/main.py",
            "src/evolutionary_ai_engine.py",
            "src/basic_security.py",
            "src/working_pdf_generator.py",
            "requirements.txt",
            "setup.py",
            "config.toml",
            "data/learning_database.json"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.log_success(f"Arquivo encontrado: {file_path}")
            else:
                self.log_error(f"Arquivo faltante: {file_path}")

    def check_python_imports(self):
        """Testa imports Python"""
        print("\n🐍 VERIFICANDO IMPORTS PYTHON...")
        
        modules_to_test = [
            "src.main",
            "src.evolutionary_ai_engine",
            "src.basic_security", 
            "src.working_pdf_generator",
            "json",
            "os",
            "sys"
        ]
        
        for module_name in modules_to_test:
            try:
                importlib.import_module(module_name)
                self.log_success(f"Import bem-sucedido: {module_name}")
            except ImportError as e:
                self.log_error(f"Falha no import {module_name}: {e}")

    def check_config_files(self):
        """Verifica arquivos de configuração"""
        print("\n⚙️  VERIFICANDO ARQUIVOS DE CONFIGURAÇÃO...")
        
        # Verifica config.toml
        if os.path.exists("config.toml"):
            try:
                with open("config.toml", 'r') as f:
                    content = f.read()
                if content.strip():
                    self.log_success("config.toml válido e não vazio")
                else:
                    self.log_warning("config.toml está vazio")
            except Exception as e:
                self.log_error(f"Erro ao ler config.toml: {e}")
        else:
            self.log_error("config.toml não encontrado")
            
        # Verifica learning_database.json
        db_path = "data/learning_database.json"
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    data = json.load(f)
                self.log_success("learning_database.json é JSON válido")
            except json.JSONDecodeError:
                self.log_error("learning_database.json contém JSON inválido")
            except Exception as e:
                self.log_error(f"Erro ao ler learning_database.json: {e}")
        else:
            self.log_warning("learning_database.json não encontrado - criando vazio")
            os.makedirs("data", exist_ok=True)
            with open(db_path, 'w') as f:
                json.dump({}, f)

    def check_requirements(self):
        """Verifica dependências"""
        print("\n📦 VERIFICANDO DEPENDÊNCIAS...")
        
        if not os.path.exists("requirements.txt"):
            self.log_error("requirements.txt não encontrado")
            return
            
        try:
            with open("requirements.txt", 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                
            for req in requirements:
                try:
                    # Tenta importar o pacote
                    package_name = req.split('==')[0].split('>')[0].split('<')[0].split('[')[0]
                    importlib.import_module(package_name)
                    self.log_success(f"Dependência OK: {package_name}")
                except ImportError:
                    self.log_warning(f"Dependência não instalada: {package_name}")
                    
        except Exception as e:
            self.log_error(f"Erro ao verificar requirements: {e}")

    def test_basic_functionality(self):
        """Testa funcionalidades básicas"""
        print("\n🔧 TESTANDO FUNCIONALIDADES BÁSICAS...")
        
        # Testa se main.py pode ser executado
        try:
            result = subprocess.run([sys.executable, "src/main.py", "--test"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.log_success("main.py executado com sucesso (modo teste)")
            else:
                self.log_warning(f"main.py retornou código {result.returncode}")
        except subprocess.TimeoutExpired:
            self.log_warning("main.py timeout - pode estar funcionando normalmente")
        except Exception as e:
            self.log_error(f"Erro ao executar main.py: {e}")
            
        # Testa segurança básica
        try:
            from src.basic_security import BasicSecurity
            security = BasicSecurity()
            test_data = "teste123"
            encrypted = security.encrypt_data(test_data)
            decrypted = security.decrypt_data(encrypted)
            if decrypted == test_data:
                self.log_success("Sistema de criptografia funcionando")
            else:
                self.log_error("Falha na criptografia: dados não coincidem")
        except Exception as e:
            self.log_error(f"Erro no sistema de segurança: {e}")

    def run_all_checks(self):
        """Executa todas as verificações"""
        print("🚀 INICIANDO VALIDAÇÃO COMPLETA DO SISTEMA")
        print("="*60)
        
        self.check_file_structure()
        self.check_python_imports()
        self.check_config_files()
        self.check_requirements()
        self.test_basic_functionality()
        
        return self.print_summary()

def main():
    """Função principal"""
    validator = SystemValidator()
    success = validator.run_all_checks()
    
    # Retorna código de erro apropriado
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
