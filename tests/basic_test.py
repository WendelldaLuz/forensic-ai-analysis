"""Teste básico do sistema"""

import sys
sys.path.append('src')

def test_imports():
    """Testa importações básicas"""
    
    print("🧪 Testando importações...")
    
    try:
        import streamlit
        print("✅ Streamlit OK")
    except ImportError:
        print("❌ Streamlit faltando")
        return False
    
    try:
        import numpy
        print("✅ NumPy OK")
    except ImportError:
        print("❌ NumPy faltando")
        return False
    
    try:
        from evolutionary_ai_engine import EnhancedForensicAI
        print("✅ IA Evolutiva OK")
        return True
    except ImportError as e:
        print(f"❌ IA Evolutiva com problema: {e}")
        return False

def test_ai_basic():
    """Teste básico da IA"""
    
    print("\n🧠 Testando IA...")
    
    try:
        from evolutionary_ai_engine import EnhancedForensicAI
        ai = EnhancedForensicAI()
        print("✅ IA inicializada")
        
        # Teste com dados simulados
        import numpy as np
        test_data = np.random.randint(-100, 100, (50, 50))
        env_params = {'temperature': 25, 'humidity': 60}
        
        result = ai.comprehensive_analysis(test_data, env_params)
        print("✅ Análise executada")
        
        if 'ra_index' in result:
            print("✅ RA-Index funcionando")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na IA: {e}")
        return False

if __name__ == "__main__":
    print("🔬 TESTE BÁSICO DO SISTEMA")
    print("=" * 40)
    
    import_ok = test_imports()
    ai_ok = test_ai_basic() if import_ok else False
    
    print(f"\n🎯 RESULTADO:")
    if import_ok and ai_ok:
        print("🎉 SISTEMA FUNCIONANDO!")
        print("🚀 Execute: streamlit run src/main.py")
    else:
        print("⚠️ Sistema tem problemas")
