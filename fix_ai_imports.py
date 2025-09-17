"""Corrige problemas de imports na IA"""

# Ler arquivo da IA
with open('src/evolutionary_ai_engine.py', 'r') as f:
    content = f.read()

# Corrigir problemas comuns
fixes = [
    # Corrigir referência a gases
    ('for gas in self.gas_model.gases:', 'for gas in self.gases:'),
    ('self.gas_model.diffusion_coefficients', 'self.diffusion_coefficients'),
    ('self.gas_model.detection_limits', 'self.detection_limits'),
    ('self.gas_model.fit_diffusion_model', 'self.fit_diffusion_model'),
    # Adicionar gases à classe EvolutionaryLearningEngine
    ('class EvolutionaryLearningEngine:', '''class EvolutionaryLearningEngine:
    """Engine de Aprendizado Evolutivo"""
    
    def __init__(self, learning_data_path: str = "data/learning_database.json"):
        # Definir gases aqui também
        self.gases = ['Putrescina', 'Cadaverina', 'Metano']
        self.diffusion_coefficients = {
            'Putrescina': 0.05,
            'Cadaverina': 0.045, 
            'Metano': 0.12
        }
        self.detection_limits = {
            'Putrescina': 5.0,
            'Cadaverina': 5.0,
            'Metano': 2.0
        }''')
]

# Aplicar correções
for old, new in fixes:
    if old in content:
        content = content.replace(old, new)
        print(f"✅ Corrigido: {old[:50]}...")

# Salvar arquivo corrigido
with open('src/evolutionary_ai_engine.py', 'w') as f:
    f.write(content)

print("🔧 IA Evolutiva corrigida!")
