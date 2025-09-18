# post_mortem_interval_model.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class PostMortemIntervalModel:
    """
    Modelo para estimativa do intervalo post-mortem baseado em
    hipóstases viscerais torácicas através de tomografia computadorizada
    """
    
    def __init__(self):
        # Parâmetros do modelo de Mitscherlich baseados no estudo
        self.alpha = 54.19  # Valor assintótico
        self.beta = 0.225   # Velocidade de crescimento
        self.alpha_se = 3.15  # Erro padrão de alpha
        self.beta_se = 0.074  # Erro padrão de beta
        
        # Locais anatômicos para análise
        self.anatomical_sites = {
            'lung_anterior': 0,    # Faixa anterior do pulmão direito
            'lung_posterior': 1,   # Faixa posterior do pulmão direito
            'atrium_ant_right': 2, # Segmento anterior do átrio direito
            'atrium_post_right': 3, # Segmento posterior do átrio direito
            'atrium_post_left': 4  # Segmento posterior do átrio esquerdo
        }
    
    def mitscherlich_model(self, t, alpha, beta):
        """
        Modelo de Mitscherlich para evolução da diferença de atenuação
        Dif42 = α * (1 - exp(-β * t))
        """
        return alpha * (1 - np.exp(-beta * t))
    
    def inverse_mitscherlich(self, dif42, alpha, beta):
        """
        Função inversa do modelo de Mitscherlich para estimar o tempo
        t = - (1/β) * ln(1 - Dif42/α)
        """
        # Prevenir valores fora do domínio
        dif42 = np.clip(dif42, 0, alpha * 0.999)  # Evitar log(0)
        return (-1 / beta) * np.log(1 - dif42 / alpha)
    
    def calculate_differences(self, attenuation_data):
        """
        Calcula as diferenças de atenuação conforme definido no estudo
        """
        differences = {
            'Dif10': attenuation_data['lung_posterior'] - attenuation_data['lung_anterior'],
            'Dif32': attenuation_data['atrium_post_right'] - attenuation_data['atrium_ant_right'],
            'Dif42': attenuation_data['atrium_post_left'] - attenuation_data['atrium_ant_right']
        }
        return differences
    
    def estimate_pmi_with_uncertainty(self, dif42):
        """
        Estima o intervalo post-mortem com cálculo de incerteza
        """
        # Estimativa pontual
        pmi_estimate = self.inverse_mitscherlich(dif42, self.alpha, self.beta)
        
        # Cálculo do erro padrão usando método delta
        # Derivadas parciais
        dtdalpha = (1 / (self.beta * (self.alpha - dif42))) * (dif42 / self.alpha)
        dtdbeta = pmi_estimate / self.beta
        
        # Variância usando método delta
        variance = (dtdalpha**2 * self.alpha_se**2 + 
                   dtdbeta**2 * self.beta_se**2 + 
                   2 * dtdalpha * dtdbeta * 0)  # Assumindo covariância zero
        
        std_error = np.sqrt(variance)
        
        # Intervalo de confiança 95%
        ci_lower = pmi_estimate - 1.96 * std_error
        ci_upper = pmi_estimate + 1.96 * std_error
        
        return {
            'pmi_estimate': pmi_estimate,
            'std_error': std_error,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper
        }
    
    def generate_simulated_data(self, n_samples=100):
        """
        Gera dados simulados baseados no estudo descrito
        """
        np.random.seed(42)
        
        data = []
        for i in range(n_samples):
            # Gerar características do caso
            true_pmi = np.random.uniform(1, 20)  # PMI real entre 1-20 horas
            has_cardiopathy = np.random.choice([0, 1], p=[0.6, 0.4])
            age = np.random.normal(55, 20)  # Idade média 55 anos
            age = max(19, min(92, age))  # Limitar entre 19-92 anos
            
            # Simular atenuações baseadas no PMI real
            base_attenuation = {
                'lung_anterior': np.random.normal(30, 5),
                'lung_posterior': np.random.normal(30, 5),
                'atrium_ant_right': np.random.normal(35, 5),
                'atrium_post_right': np.random.normal(35, 5),
                'atrium_post_left': np.random.normal(35, 5)
            }
            
            # Aplicar efeito do PMI (aumento progressivo das diferenças)
            time_effect = self.mitscherlich_model(true_pmi, self.alpha, self.beta)
            
            attenuation_data = {
                'lung_anterior': base_attenuation['lung_anterior'] - time_effect * 0.3,
                'lung_posterior': base_attenuation['lung_posterior'] + time_effect * 0.7,
                'atrium_ant_right': base_attenuation['atrium_ant_right'] - time_effect * 0.4,
                'atrium_post_right': base_attenuation['atrium_post_right'] + time_effect * 0.6,
                'atrium_post_left': base_attenuation['atrium_post_left'] + time_effect * 0.8
            }
            
            # Adicionar ruído
            for key in attenuation_data:
                attenuation_data[key] += np.random.normal(0, 2)
            
            # Calcular diferenças
            differences = self.calculate_differences(attenuation_data)
            
            # Estimar PMI
            pmi_estimation = self.estimate_pmi_with_uncertainty(differences['Dif42'])
            
            data.append({
                'case_id': i + 1,
                'true_pmi': true_pmi,
                'estimated_pmi': pmi_estimation['pmi_estimate'],
                'pmi_std_error': pmi_estimation['std_error'],
                'pmi_ci_lower': pmi_estimation['ci_95_lower'],
                'pmi_ci_upper': pmi_estimation['ci_95_upper'],
                'has_cardiopathy': has_cardiopathy,
                'age': age,
                'Dif10': differences['Dif10'],
                'Dif32': differences['Dif32'],
                'Dif42': differences['Dif42'],
                **attenuation_data
            })
        
        return pd.DataFrame(data)
    
    def evaluate_model(self, data):
        """
        Avalia o desempenho do modelo
        """
        # Métricas de performance
        mse = mean_squared_error(data['true_pmi'], data['estimated_pmi'])
        rmse = np.sqrt(mse)
        r2 = r2_score(data['true_pmi'], data['estimated_pmi'])
        
        # Erro absoluto médio
        absolute_errors = np.abs(data['true_pmi'] - data['estimated_pmi'])
        mae = np.mean(absolute_errors)
        
        # Precisão dentro de intervalos
        within_1h = np.mean(absolute_errors <= 1)
        within_2h = np.mean(absolute_errors <= 2)
        within_3h = np.mean(absolute_errors <= 3)
        
        # Correlações
        corr_dif42 = stats.pearsonr(data['true_pmi'], data['Dif42'])[0]
        corr_estimate = stats.pearsonr(data['true_pmi'], data['estimated_pmi'])[0]
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'within_1h': within_1h,
            'within_2h': within_2h,
            'within_3h': within_3h,
            'corr_dif42': corr_dif42,
            'corr_estimate': corr_estimate
        }
    
    def visualize_results(self, data, evaluation_results, save_path='results/pmi_analysis.png'):
        """
        Cria visualizações completas dos resultados
        """
        plt.figure(figsize=(20, 15))
        
        # 1. Relação entre PMI real e estimado
        plt.subplot(3, 3, 1)
        plt.scatter(data['true_pmi'], data['estimated_pmi'], alpha=0.6)
        plt.plot([0, 20], [0, 20], 'r--', label='Linha de perfeita concordância')
        plt.xlabel('PMI Real (horas)')
        plt.ylabel('PMI Estimado (horas)')
        plt.title(f'Relação PMI Real vs Estimado\nR² = {evaluation_results["r2"]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Evolução das diferenças de atenuação com o tempo
        plt.subplot(3, 3, 2)
        time_points = np.linspace(0, 20, 100)
        dif42_curve = self.mitscherlich_model(time_points, self.alpha, self.beta)
        
        plt.plot(time_points, dif42_curve, 'b-', label='Modelo de Mitscherlich', linewidth=2)
        plt.scatter(data['true_pmi'], data['Dif42'], alpha=0.6, color='orange', label='Dados simulados')
        plt.xlabel('PMI (horas)')
        plt.ylabel('Dif42 (UH)')
        plt.title('Evolução da Dif42 com o Tempo Post-Mortem')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Distribuição dos erros de estimativa
        plt.subplot(3, 3, 3)
        errors = data['estimated_pmi'] - data['true_pmi']
        plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Erro de Estimativa (horas)')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição dos Erros\nMAE = {evaluation_results["mae"]:.2f} horas')
        plt.grid(True, alpha=0.3)
        
        # 4. Comparação entre casos com e sem cardiopatia
        plt.subplot(3, 3, 4)
        cardiopathy_data = data[data['has_cardiopathy'] == 1]
        no_cardiopathy_data = data[data['has_cardiopathy'] == 0]
        
        plt.boxplot([no_cardiopathy_data['estimated_pmi'] - no_cardiopathy_data['true_pmi'],
                    cardiopathy_data['estimated_pmi'] - cardiopathy_data['true_pmi']],
                   labels=['Sem Cardiopatia', 'Com Cardiopatia'])
        plt.ylabel('Erro de Estimativa (horas)')
        plt.title('Erro de Estimativa por Presença de Cardiopatia')
        plt.grid(True, alpha=0.3)
        
        # 5. Precisão por faixa de PMI
        plt.subplot(3, 3, 5)
        pmi_bins = [0, 6, 12, 18, 24]
        bin_labels = ['0-6h', '6-12h', '12-18h', '18-24h']
        
        mae_by_bin = []
        for i in range(len(pmi_bins) - 1):
            bin_mask = (data['true_pmi'] >= pmi_bins[i]) & (data['true_pmi'] < pmi_bins[i + 1])
            if bin_mask.any():
                bin_errors = np.abs(data.loc[bin_mask, 'estimated_pmi'] - data.loc[bin_mask, 'true_pmi'])
                mae_by_bin.append(bin_errors.mean())
            else:
                mae_by_bin.append(0)
        
        plt.bar(bin_labels, mae_by_bin)
        plt.xlabel('Faixa de PMI (horas)')
        plt.ylabel('Erro Absoluto Médio (horas)')
        plt.title('Precisão por Faixa de Intervalo Post-Mortem')
        plt.grid(True, alpha=0.3)
        
        # 6. Intervalos de confiança
        plt.subplot(3, 3, 6)
        sample_cases = data.head(10).copy()
        sample_cases = sample_cases.sort_values('true_pmi')
        
        for i, (idx, row) in enumerate(sample_cases.iterrows()):
            plt.errorbar(row['estimated_pmi'], i, 
                        xerr=[[row['estimated_pmi'] - row['pmi_ci_lower']], 
                              [row['pmi_ci_upper'] - row['estimated_pmi']]],
                        fmt='o', capsize=5)
            plt.plot(row['true_pmi'], i, 'rx', markersize=10, label='Real' if i == 0 else "")
        
        plt.yticks(range(len(sample_cases)), [f'Caso {i+1}' for i in range(len(sample_cases))])
        plt.xlabel('PMI (horas)')
        plt.title('Intervalos de Confiança das Estimativas\n(10 primeiros casos)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Correlação entre todas as diferenças
        plt.subplot(3, 3, 7)
        differences = data[['Dif10', 'Dif32', 'Dif42']]
        correlation_matrix = differences.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Correlação entre Diferenças de Atenuação')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return evaluation_results

# Função para integração com o sistema existente
def integrate_pmi_model():
    """
    Integra o modelo de estimativa de PMI com o sistema existente
    """
    print("Desenvolvendo modelo de estimativa do intervalo post-mortem...")
    
    # Inicializar modelo
    pmi_model = PostMortemIntervalModel()
    
    # Gerar dados simulados
    print("Gerando dados simulados...")
    pmi_data = pmi_model.generate_simulated_data(n_samples=200)
    
    # Avaliar modelo
    print("Avaliando desempenho do modelo...")
    evaluation_results = pmi_model.evaluate_model(pmi_data)
    
    # Visualizar resultados
    print("Criando visualizações...")
    pmi_model.visualize_results(pmi_data, evaluation_results)
    
    # Salvar resultados
    pmi_data.to_csv('results/pmi_estimation_data.csv', index=False)
    
    print(f"\nResultados do Modelo de Estimativa de PMI:")
    print(f"Erro Absoluto Médio: {evaluation_results['mae']:.2f} horas")
    print(f"R²: {evaluation_results['r2']:.3f}")
    print(f"Precisão dentro de 1 hora: {evaluation_results['within_1h']*100:.1f}%")
    print(f"Precisão dentro de 2 horas: {evaluation_results['within_2h']*100:.1f}%")
    print(f"Precisão dentro de 3 horas: {evaluation_results['within_3h']*100:.1f}%")
    print(f"Correlação Dif42-PMI: {evaluation_results['corr_dif42']:.3f}")
    
    return {
        'pmi_data': pmi_data,
        'evaluation_results': evaluation_results,
        'model_parameters': {
            'alpha': pmi_model.alpha,
            'beta': pmi_model.beta,
            'alpha_se': pmi_model.alpha_se,
            'beta_se': pmi_model.beta_se
        }
    }

if __name__ == "__main__":
    integrate_pmi_model()
