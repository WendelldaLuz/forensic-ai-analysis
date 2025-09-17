"""
Script para corrigir erro na IA Evolutiva
"""

# Ler arquivo atual
with open('src/evolutionary_ai_engine.py', 'r') as f:
    content = f.read()

# Procurar e corrigir o erro
if "self.gas_model.diffusion_coefficients[gas]" in content:
    # O erro está na linha onde chama self.gas_model.gases
    # mas deveria ser self.gases
    
    content = content.replace(
        "for gas in self.gas_model.gases:",
        "for gas in self.gases:"
    )
    
    content = content.replace(
        "self.gas_model.diffusion_coefficients[gas]",
        "self.diffusion_coefficients[gas]"
    )
    
    content = content.replace(
        "self.gas_model.detection_limits[gas]",
        "self.detection_limits[gas]"
    )
    
    # Salvar correção
    with open('src/evolutionary_ai_engine.py', 'w') as f:
        f.write(content)
    
    print("✅ Erro na IA corrigido!")
else:
    print("ℹ️ Erro não encontrado ou já corrigido")
