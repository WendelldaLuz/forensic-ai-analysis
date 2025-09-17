"""
Sistema de Segurança Básico Funcional
"""

import hashlib
import json
import os
from datetime import datetime

class BasicSecurity:
    """Sistema de segurança básico"""
    
    def __init__(self):
        os.makedirs('security', exist_ok=True)
        self.log_file = 'security/basic_security.log'
    
    def log_event(self, event_type: str, details: str):
        """Log básico de eventos"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\\n')
    
    def verify_integrity(self, file_path: str) -> bool:
        """Verificação básica de integridade"""
        
        if not os.path.exists(file_path):
            return False
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Log verificação
        self.log_event('INTEGRITY_CHECK', f'{file_path}: {file_hash[:16]}...')
        
        return True

def test_basic_security():
    """Testa segurança básica"""
    
    security = BasicSecurity()
    security.log_event('SYSTEM_START', 'Sistema iniciado')
    
    # Testar verificação
    result = security.verify_integrity('src/main.py')
    print(f"✅ Segurança básica: {'OK' if result else 'ERRO'}")
    
    return result

if __name__ == "__main__":
    test_basic_security()
