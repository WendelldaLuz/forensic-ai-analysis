#!/usr/bin/env python3
"""
Verifica arquivos vazios ou com conteúdo genérico
"""

import os
import json

def check_empty_or_generic():
    """Verifica arquivos vazios ou genéricos"""
    
    print("🔍 VERIFICANDO ARQUIVOS VAZIOS OU GENÉRICOS")
    print("=" * 50)
    
    # Encontrar todos os arquivos
    all_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if not file.startswith('.') and not file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    empty_files = []
    small_files = []
    generic_files = []
    
    for file_path in all_files:
        try:
            size = os.path.getsize(file_path)
            
            if size == 0:
                empty_files.append(file_path)
                print(f"📭 VAZIO: {file_path} (0 bytes)")
                
            elif size < 100:
                small_files.append(file_path)
                print(f"⚠️ PEQUENO: {file_path} ({size} bytes)")
                
                # Verificar conteúdo
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if not content or content in ['', '# TODO', 'pass', 'import os']:
                            generic_files.append(file_path)
                            print(f"   📝 Conteúdo genérico: '{content}'")
                except:
                    pass
            
            else:
                print(f"✅ OK: {file_path} ({size} bytes)")
                
        except Exception as e:
            print(f"❌ ERRO: {file_path} - {e}")
    
    # Relatório final
    print(f"\n📊 RESUMO:")
    print(f"📭 Arquivos vazios: {len(empty_files)}")
    print(f"⚠️ Arquivos pequenos: {len(small_files)}")
    print(f"📝 Arquivos genéricos: {len(generic_files)}")
    
    if empty_files or generic_files:
        print(f"\n🔧 ARQUIVOS QUE PRECISAM DE CONTEÚDO:")
        for file in empty_files + generic_files:
            print(f"  • {file}")
    
    return empty_files, small_files, generic_files

if __name__ == "__main__":
    check_empty_or_generic()
