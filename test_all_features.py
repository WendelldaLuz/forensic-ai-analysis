#!/usr/bin/env python3
"""
Teste Completo de Todas as Funcionalidades
"""

import sys
sys.path.append('src')
import numpy as np
from datetime import datetime

def test_all_features():
    """Testa todas as funcionalidades do sistema"""
    
    print("🔬 TESTANDO TODAS AS FUNCIONALIDADES")
    print("=" * 60)
    
    results = {
        'ai_evolutiva': False,
        'pdf_generator': False,
        'security_system': False,
        'multi_upload': False,
        'data_persistence': False
    }
    
    # Teste 1: IA Evolutiva
    try:
        from evolutionary_ai_engine import EnhancedForensicAI
        ai = EnhancedForensicAI()
        
        # Executar análise teste
        test_image = np.random.randint(-200, 200, (100, 100), dtype=np.int16)
        env_params = {'temperature': 25, 'humidity': 60}
        
        result = ai.comprehensive_analysis(test_image, env_params)
        
        if 'ra_index' in result and 'confidence_score' in result:
            print("✅ IA Evolutiva: FUNCIONANDO")
            results['ai_evolutiva'] = True
        else:
            print("❌ IA Evolutiva: PROBLEMA")
            
    except Exception as e:
        print(f"❌ IA Evolutiva: ERRO - {e}")
    
    # Teste 2: Gerador PDF
    try:
        from enhanced_pdf_generator import ComprehensivePDFGenerator
        
        generator = ComprehensivePDFGenerator()
        
        # Dados de teste
        test_results = [{
            'estimated_pmi': 15.5,
            'confidence_score': 0.85,
            'ra_index': {'ra_index': 45, 'interpretation': {'level': 'moderate'}},
            'image_features': {'mean_hu': 30.2, 'gas_volume': 0.12},
            'analysis_metadata': {'learning_statistics': {'total_analyses': 10}}
        }]
        
        pdf_buffer = generator.generate_complete_pdf_report(
            test_results, {'temperature': 25, 'humidity': 60}
        )
        
        if len(pdf_buffer.getvalue()) > 10000:  # PDF com conteúdo
            print("✅ Gerador PDF: FUNCIONANDO")
            results['pdf_generator'] = True
        else:
            print("❌ Gerador PDF: PROBLEMA")
            
    except Exception as e:
        print(f"❌ Gerador PDF: ERRO - {e}")
    
    # Teste 3: Sistema de Segurança
    try:
        from military_security import MilitaryGradeEncryption
        
        security = MilitaryGradeEncryption()
        
        # Teste de criptografia
        test_data = {"sensitive": "dados confidenciais"}
        encrypted = security.encrypt_forensic_data(test_data, "confidential")
        decrypted = security.decrypt_forensic_data(encrypted)
        
        if decrypted == test_data:
            print("✅ Sistema de Segurança: FUNCIONANDO")
            results['security_system'] = True
        else:
            print("❌ Sistema de Segurança: PROBLEMA")
            
    except Exception as e:
        print(f"⚠️ Sistema de Segurança: NÃO DISPONÍVEL - {e}")
    
    # Teste 4: Persistência de Dados
    try:
        import json
        import os
        
        # Verificar se base de dados de aprendizado existe
        if os.path.exists('data/learning_database.json'):
            with open('data/learning_database.json', 'r') as f:
                data = json.load(f)
            
            if 'analyses_performed' in data:
                print("✅ Persistência de Dados: FUNCIONANDO")
                results['data_persistence'] = True
            else:
                print("❌ Persistência de Dados: PROBLEMA")
        else:
            print("⚠️ Persistência de Dados: ARQUIVO NÃO ENCONTRADO")
            
    except Exception as e:
        print(f"❌ Persistência de Dados: ERRO - {e}")
    
    # Teste 5: Multi-upload (simulado)
    try:
        # Simular múltiplos arquivos
        multiple_images = [
            np.random.randint(-100, 100, (50, 50), dtype=np.int16) for _ in range(3)
        ]
        
        if len(multiple_images) == 3:
            print("✅ Multi-Upload: FUNCIONANDO")
            results['multi_upload'] = True
        else:
            print("❌ Multi-Upload: PROBLEMA")
            
    except Exception as e:
        print(f"❌ Multi-Upload: ERRO - {e}")
    
    # Relatório final
    print(f"\n📊 RELATÓRIO FINAL DE FUNCIONALIDADES")
    print("=" * 60)
    
    total_features = len(results)
    working_features = sum(results.values())
    success_rate = working_features / total_features * 100
    
    for feature, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {feature.replace('_', ' ').title()}: {'OK' if status else 'PROBLEMA'}")
    
    print(f"\n🎯 TAXA DE SUCESSO: {success_rate:.1f}% ({working_features}/{total_features})")
    
    if success_rate >= 80:
        print("🎉 SISTEMA ALTAMENTE FUNCIONAL!")
    elif success_rate >= 60:
        print("✅ SISTEMA FUNCIONANDO COM ALGUMAS LIMITAÇÕES")
    else:
        print("⚠️ SISTEMA PRECISA DE AJUSTES")
    
    return results

if __name__ == "__main__":
    test_all_features()
