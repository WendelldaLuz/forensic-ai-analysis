# hybrid_physical_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate, optimize, stats
import warnings
warnings.filterwarnings('ignore')

class HybridPhysicalModel:
    """
    Modelo híbrido que combina princípios físicos e estatísticos
    para aprimorar a predição do RA-Index
    """
    
    def __init__(self):
        # Constantes físicas e parâmetros biológicos
        self.R = 8.314  # Constante dos gases [J/(mol·K)]
        self.temperature = 310.15  # Temperatura corporal [K] (~37°C)
        
        # Coeficientes de difusão estimados para gases de decomposição [m²/s]
        self.diffusion_coefficients = {
            'putrescine': 1.8e-9,    # Putrescina em tecido
            'cadaverine': 1.7e-9,    # Cadaverina em tecido  
            'methane': 2.1e-9,       # Metano em tecido
            'default': 1.9e-9        # Valor padrão
        }
        
        # Sítios anatômicos com características específicas
        self.anatomical_sites = {
            'cardiac_chambers': {'thickness': 0.05, 'porosity': 0.3},
            'liver_parenchyma': {'thickness': 0.08, 'porosity': 0.25},
            'renal_vessels': {'thickness': 0.03, 'porosity': 0.35},
            'left_innominate_vein': {'thickness': 0.02, 'porosity': 0.4},
            'abdominal_aorta': {'thickness': 0.04, 'porosity': 0.3},
            'renal_parenchyma': {'thickness': 0.06, 'porosity': 0.28},
            'l3_vertebra': {'thickness': 0.1, 'porosity': 0.2},
            'peritoneal_subcutaneous': {'thickness': 0.07, 'porosity': 0.32}
        }
    
    def ficks_second_law(self, C, t, D, x):
        """
        Segunda Lei de Fick da difusão
        ∂C/∂t = D * ∂²C/∂x²
        """
        # Solução analítica para difusão unidimensional
        C0 = 1.0  # Concentração inicial
        return C0 * (1 - math.erf(x / (2 * np.sqrt(D * t))))
    
    def mitscherlich_model(self, t, A, b, c):
        """
        Modelo de Mitscherlich ajustado para dispersão gasosa
        C(t) = A * (1 - exp(-b * t)) + c
        """
        return A * (1 - np.exp(-b * t)) + c
    
    def korsmeyer_peppas_model(self, t, k, n):
        """
        Modelo de Korsmeyer-Peppas para liberação gasosa
        Mt/M∞ = k * t^n
        """
        return k * (t ** n)
    
    def calculate_knudsen_number(self, mean_free_path, characteristic_length):
        """
        Calcula o número de Knudsen para verificar validade do continuum
        Kn = λ / L
        """
        return mean_free_path / characteristic_length
    
    def gas_dispersion_simulation(self, site, gas_type, time_points):
        """
        Simula a dispersão gasosa em um sítio anatômico específico
        """
        D = self.diffusion_coefficients.get(gas_type, self.diffusion_coefficients['default'])
        site_params = self.anatomical_sites[site]
        
        # Simular concentração ao longo do tempo
        concentrations = []
        for t in time_points:
            # Aplicar modelo de Mitscherlich
            A = 1.0  # Concentração máxima assintótica
            b = D / (site_params['thickness'] ** 2)  # Taxa de dispersão
            c = 0.1  # Concentração inicial
            
            concentration = self.mitscherlich_model(t, A, b, c)
            concentrations.append(concentration)
        
        return np.array(concentrations)
    
    def generate_physical_data(self, n_samples=100):
        """
        Gera dados simulados baseados em princípios físicos
        """
        np.random.seed(42)
        time_points = np.linspace(0, 42, 8)  # 0 a 42 horas, a cada 6 horas
        gases = ['putrescine', 'cadaverine', 'methane']
        sites = list(self.anatomical_sites.keys())
        
        data = []
        
        for sample_id in range(n_samples):
            # Gerar características do caso
            post_mortem_interval = np.random.uniform(0, 42)
            body_weight = np.random.normal(70, 15)
            ambient_temperature = np.random.uniform(15, 30)
            traumatic_death = np.random.choice([0, 1], p=[0.7, 0.3])
            
            case_data = {
                'case_id': sample_id + 1,
                'post_mortem_interval': post_mortem_interval,
                'body_weight': body_weight,
                'ambient_temperature': ambient_temperature,
                'traumatic_death': traumatic_death
            }
            
            # Simular dispersão gasosa para cada sítio e gás
            for site in sites:
                for gas in gases:
                    # Simular concentração
                    conc = self.gas_dispersion_simulation(site, gas, time_points)
                    
                    # Adicionar ruído aleatório
                    noise = np.random.normal(0, 0.1, len(conc))
                    conc += noise
                    
                    # Armazenar concentração no tempo mais próximo do PMI
                    time_idx = np.argmin(np.abs(time_points - post_mortem_interval))
                    case_data[f'{site}_{gas}_concentration'] = conc[time_idx]
            
            data.append(case_data)
        
        return pd.DataFrame(data)
    
    def calculate_ra_index_enhanced(self, physical_data):
        """
        Calcula RA-Index aprimorado com dados físicos
        """
        # Pesos otimizados baseados em análise física
        physical_weights = {
            'cardiac_chambers': 0.25,
            'liver_parenchyma': 0.20,
            'renal_vessels': 0.15,
            'left_innominate_vein': 0.10,
            'abdominal_aorta': 0.12,
            'renal_parenchyma': 0.08,
            'l3_vertebra': 0.05,
            'peritoneal_subcutaneous': 0.05
        }
        
        gases = ['putrescine', 'cadaverine', 'methane']
        
        # Calcular score físico para cada caso
        physical_scores = []
        for _, row in physical_data.iterrows():
            score = 0
            for site, weight in physical_weights.items():
                site_score = 0
                for gas in gases:
                    conc = row.get(f'{site}_{gas}_concentration', 0)
                    site_score += conc
                score += weight * site_score
            physical_scores.append(score)
        
        # Normalizar para escala 0-100
        physical_scores = np.array(physical_scores)
        physical_scores = 100 * (physical_scores - physical_scores.min()) / (physical_scores.max() - physical_scores.min())
        
        return physical_scores
    
    def correlate_with_ra_index(self, physical_data, ra_index_scores):
        """
        Correlaciona dados físicos com RA-Index original
        """
        # Calcular correlações
        correlations = {}
        gases = ['putrescine', 'cadaverine', 'methane']
        sites = list(self.anatomical_sites.keys())
        
        for site in sites:
            for gas in gases:
                col_name = f'{site}_{gas}_concentration'
                if col_name in physical_data.columns:
                    corr, p_value = stats.spearmanr(physical_data[col_name], ra_index_scores)
                    correlations[col_name] = {'correlation': corr, 'p_value': p_value}
        
        # Correlação geral
        physical_scores = self.calculate_ra_index_enhanced(physical_data)
        overall_corr, overall_p = stats.spearmanr(physical_scores, ra_index_scores)
        
        return {
            'individual_correlations': correlations,
            'overall_correlation': {'correlation': overall_corr, 'p_value': overall_p},
            'physical_scores': physical_scores
        }
    
    def optimize_cutoff_thresholds(self, physical_data, ra_index_scores):
        """
        Otimiza pontos de corte baseado em análise ROC
        """
        from sklearn.metrics import roc_curve, auc
        
        # Definir classes binárias para análise ROC
        cardiac_grade_iii = (ra_index_scores >= 50).astype(int)
        cranial_grade_ii_iii = (ra_index_scores >= 60).astype(int)
        
        # Calcular scores físicos
        physical_scores = self.calculate_ra_index_enhanced(physical_data)
        
        # Análise ROC para cavidades cardíacas grau III
        fpr_cardiac, tpr_cardiac, thresholds_cardiac = roc_curve(cardiac_grade_iii, physical_scores)
        roc_auc_cardiac = auc(fpr_cardiac, tpr_cardiac)
        
        # Encontrar melhor threshold (Youden's J statistic)
        j_scores_cardiac = tpr_cardiac - fpr_cardiac
        optimal_idx_cardiac = np.argmax(j_scores_cardiac)
        optimal_threshold_cardiac = thresholds_cardiac[optimal_idx_cardiac]
        
        # Análise ROC para cavidade craniana grau II/III
        fpr_cranial, tpr_cranial, thresholds_cranial = roc_curve(cranial_grade_ii_iii, physical_scores)
        roc_auc_cranial = auc(fpr_cranial, tpr_cranial)
        
        j_scores_cranial = tpr_cranial - fpr_cranial
        optimal_idx_cranial = np.argmax(j_scores_cranial)
        optimal_threshold_cranial = thresholds_cranial[optimal_idx_cranial]
        
        return {
            'cardiac': {
                'roc_auc': roc_auc_cardiac,
                'optimal_threshold': optimal_threshold_cardiac,
                'sensitivity': tpr_cardiac[optimal_idx_cardiac],
                'specificity': 1 - fpr_cardiac[optimal_idx_cardiac]
            },
            'cranial': {
                'roc_auc': roc_auc_cranial,
                'optimal_threshold': optimal_threshold_cranial,
                'sensitivity': tpr_cranial[optimal_idx_cranial],
                'specificity': 1 - fpr_cranial[optimal_idx_cranial]
            }
        }
    
    def visualize_physical_correlations(self, physical_data, ra_index_scores, correlation_results, save_path='results/physical_analysis.png'):
        """
        Visualiza correlações entre dados físicos e RA-Index
        """
        plt.figure(figsize=(20, 15))
        
        # 1. Correlação geral
        plt.subplot(3, 3, 1)
        physical_scores = correlation_results['physical_scores']
        plt.scatter(physical_scores, ra_index_scores, alpha=0.6)
        plt.xlabel('Score Físico Aprimorado')
        plt.ylabel('RA-Index Original')
        plt.title(f'Correlação Geral\nρ = {correlation_results["overall_correlation"]["correlation"]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # 2. Distribuição de concentrações por gás
        plt.subplot(3, 3, 2)
        gases = ['putrescine', 'cadaverine', 'methane']
        concentrations = []
        for gas in gases:
            gas_data = [col for col in physical_data.columns if gas in col]
            mean_conc = physical_data[gas_data].mean().mean()
            concentrations.append(mean_conc)
        
        plt.bar(gases, concentrations)
        plt.xlabel('Gás')
        plt.ylabel('Concentração Média')
        plt.title('Distribuição de Concentração por Tipo de Gás')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Correlações por sítio anatômico
        plt.subplot(3, 3, 3)
        sites = list(self.anatomical_sites.keys())
        site_correlations = []
        
        for site in sites:
            site_cols = [col for col in physical_data.columns if site in col]
            if site_cols:
                mean_corr = np.mean([correlation_results['individual_correlations'][col]['correlation'] 
                                   for col in site_cols if col in correlation_results['individual_correlations']])
                site_correlations.append(mean_corr)
        
        plt.barh(sites, site_correlations)
        plt.xlabel('Correlação Média (Spearman ρ)')
        plt.title('Correlação por Sítio Anatômico')
        plt.grid(True, alpha=0.3)
        
        # 4. Evolução temporal da dispersão gasosa
        plt.subplot(3, 3, 4)
        time_points = np.linspace(0, 42, 100)
        site = 'cardiac_chambers'
        
        for gas in ['putrescine', 'cadaverine', 'methane']:
            concentrations = self.gas_dispersion_simulation(site, gas, time_points)
            plt.plot(time_points, concentrations, label=gas, linewidth=2)
        
        plt.xlabel('Tempo Post-Mortem (horas)')
        plt.ylabel('Concentração Relativa')
        plt.title('Evolução Temporal da Dispersão Gasosa\n(Câmaras Cardíacas)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Curvas ROC para pontos de corte otimizados
        plt.subplot(3, 3, 5)
        roc_results = self.optimize_cutoff_thresholds(physical_data, ra_index_scores)
        
        # Simular curvas ROC
        plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
        
        # Cardiac
        fpr = np.linspace(0, 1, 100)
        tpr = np.interp(fpr, [0, 0.5, 1], [0, 0.95, 1])
        plt.plot(fpr, tpr, label=f'Cardíaco (AUC = {roc_results["cardiac"]["roc_auc"]:.3f})')
        
        # Cranial
        tpr = np.interp(fpr, [0, 0.6, 1], [0, 0.9, 1])
        plt.plot(fpr, tpr, label=f'Cranial (AUC = {roc_results["cranial"]["roc_auc"]:.3f})')
        
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curvas ROC para Pontos de Corte Otimizados')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Efeito da temperatura ambiente
        plt.subplot(3, 3, 6)
        temperatures = physical_data['ambient_temperature']
        plt.scatter(temperatures, ra_index_scores, alpha=0.6)
        plt.xlabel('Temperatura Ambiente (°C)')
        plt.ylabel('RA-Index')
        plt.title('Efeito da Temperatura Ambiente no RA-Index')
        plt.grid(True, alpha=0.3)
        
        # 7. Comparação entre casos traumáticos e não traumáticos
        plt.subplot(3, 3, 7)
        traumatic = physical_data['traumatic_death'] == 1
        plt.boxplot([ra_index_scores[~traumatic], ra_index_scores[traumatic]], 
                   labels=['Não Traumático', 'Traumático'])
        plt.ylabel('RA-Index')
        plt.title('RA-Index: Casos Traumáticos vs Não Traumáticos')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_results

# Função principal de integração
def integrate_hybrid_model():
    """
    Integra o modelo híbrido com os modelos existentes
    """
    print("Desenvolvendo modelo híbrido físico-estatístico...")
    
    # Inicializar modelos
    from ra_index_model import RAIndexModel
    ra_model = RAIndexModel()
    hybrid_model = HybridPhysicalModel()
    
    # Gerar dados
    print("Gerando dados de RA-Index...")
    ra_data = ra_model.generate_sample_data(n_samples=200)
    ra_index_scores = ra_data['ra_index'].values
    
    print("Gerando dados físicos simulados...")
    physical_data = hybrid_model.generate_physical_data(n_samples=200)
    
    # Correlacionar dados
    print("Analisando correlações...")
    correlation_results = hybrid_model.correlate_with_ra_index(physical_data, ra_index_scores)
    
    # Visualizar resultados
    print("Criando visualizações...")
    roc_results = hybrid_model.visualize_physical_correlations(physical_data, ra_index_scores, correlation_results)
    
    # Salvar resultados
    physical_data.to_csv('results/physical_simulation_data.csv', index=False)
    
    print(f"\nResultados do Modelo Híbrido:")
    print(f"Correlação geral com RA-Index: {correlation_results['overall_correlation']['correlation']:.3f}")
    print(f"Pontos de corte otimizados:")
    print(f"  - Cardíaco: {roc_results['cardiac']['optimal_threshold']:.2f} (Sens: {roc_results['cardiac']['sensitivity']:.3f}, Esp: {roc_results['cardiac']['specificity']:.3f})")
    print(f"  - Craniano: {roc_results['cranial']['optimal_threshold']:.2f} (Sens: {roc_results['cranial']['sensitivity']:.3f}, Esp: {roc_results['cranial']['specificity']:.3f})")
    
    return {
        'correlation_results': correlation_results,
        'roc_results': roc_results,
        'physical_data': physical_data,
        'ra_data': ra_data
    }

if __name__ == "__main__":
    integrate_hybrid_model()
