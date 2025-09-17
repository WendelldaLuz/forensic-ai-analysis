"""
Gerador PDF Funcional Simplificado
"""

import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
from typing import Dict, List

class WorkingPDFGenerator:
    """Gerador PDF que funciona garantidamente"""
    
    def __init__(self):
        plt.style.use('default')
        
    def generate_pdf_report(self, analysis_results: List[Dict], 
                           environmental_params: Dict) -> io.BytesIO:
        """Gera relatório PDF funcional"""
        
        buffer = io.BytesIO()
        
        with PdfPages(buffer) as pdf:
            # Página 1: Resumo
            self._create_summary_page(pdf, analysis_results, environmental_params)
            
            # Página 2: Gráficos
            self._create_charts_page(pdf, analysis_results)
            
            # Página 3: Dados
            self._create_data_page(pdf, analysis_results)
        
        buffer.seek(0)
        return buffer
    
    def _create_summary_page(self, pdf, results, env_params):
        """Página de resumo"""
        
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        # Título
        ax.text(0.5, 0.9, 'RELATÓRIO DE ANÁLISE FORENSE', 
               ha='center', fontsize=20, fontweight='bold')
        
        ax.text(0.5, 0.85, 'Sistema de IA Evolutiva', 
               ha='center', fontsize=16)
        
        # Informações
        info_text = f"""
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Casos Analisados: {len([r for r in results if 'error' not in r])}
Temperatura: {env_params.get('temperature', 25)}°C
Umidade: {env_params.get('humidity', 60)}%

RESULTADOS PRINCIPAIS:
"""
        
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            avg_pmi = np.mean([r.get('estimated_pmi', 0) for r in valid_results])
            avg_confidence = np.mean([r.get('confidence_score', 0) for r in valid_results])
            
            info_text += f"""
IPM Médio Estimado: {avg_pmi:.1f} horas
Confiança Média: {avg_confidence:.1%}
Status: Sistema Funcionando
"""
        
        ax.text(0.1, 0.7, info_text, fontsize=12, verticalalignment='top')
        
        pdf.savefig(fig)
        plt.close()
    
    def _create_charts_page(self, pdf, results):
        """Página de gráficos"""
        
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11.69, 8.27))
        
        # Gráfico 1: IPM por caso
        pmi_values = [r.get('estimated_pmi', 0) for r in valid_results]
        cases = [f'Caso {i+1}' for i in range(len(pmi_values))]
        
        ax1.bar(cases, pmi_values, color='skyblue')
        ax1.set_title('IPM por Caso')
        ax1.set_ylabel('IPM (horas)')
        
        # Gráfico 2: RA-Index
        ra_values = [r.get('ra_index', {}).get('ra_index', 0) for r in valid_results]
        ax2.bar(cases, ra_values, color='orange')
        ax2.set_title('RA-Index por Caso')
        ax2.set_ylabel('RA-Index')
        
        # Gráfico 3: Confiança
        confidence_values = [r.get('confidence_score', 0) for r in valid_results]
        ax3.bar(cases, confidence_values, color='green')
        ax3.set_title('Confiança por Caso')
        ax3.set_ylabel('Confiança')
        
        # Gráfico 4: Correlação
        if len(pmi_values) > 1:
            ax4.scatter(pmi_values, ra_values, c=confidence_values, s=100)
            ax4.set_title('IPM vs RA-Index')
            ax4.set_xlabel('IPM (horas)')
            ax4.set_ylabel('RA-Index')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_data_page(self, pdf, results):
        """Página de dados"""
        
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'DADOS DETALHADOS', 
               ha='center', fontsize=18, fontweight='bold')
        
        data_text = "ANÁLISE POR CASO:\\n\\n"
        
        for i, result in enumerate(results):
            if 'error' not in result:
                data_text += f"CASO {i+1}:\\n"
                data_text += f"• IPM: {result.get('estimated_pmi', 0):.1f}h\\n"
                data_text += f"• RA-Index: {result.get('ra_index', {}).get('ra_index', 0):.1f}\\n"
                data_text += f"• Confiança: {result.get('confidence_score', 0):.1%}\\n\\n"
        
        ax.text(0.1, 0.8, data_text, fontsize=11, verticalalignment='top')
        
        pdf.savefig(fig)
        plt.close()

# Função de teste
def test_working_pdf():
    """Testa gerador PDF funcional"""
    
    generator = WorkingPDFGenerator()
    
    test_results = [{
        'estimated_pmi': 20.5,
        'confidence_score': 0.88,
        'ra_index': {'ra_index': 55},
    }]
    
    env_params = {'temperature': 23, 'humidity': 65}
    
    pdf_buffer = generator.generate_pdf_report(test_results, env_params)
    
    with open('test_working_report.pdf', 'wb') as f:
        f.write(pdf_buffer.getvalue())
    
    print("✅ PDF funcional gerado: test_working_report.pdf")
    return pdf_buffer

if __name__ == "__main__":
    test_working_pdf()
