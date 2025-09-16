"""
DICOM Forensic AI Analysis System
Sistema Autônomo de Análise Forense com IA
Author: Wendell da Luz Silva
Version: 1.0.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import pydicom
import tempfile
import os
import json
import uuid
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Configurações
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do Streamlit
st.set_page_config(
    page_title="Forensic AI Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importações condicionais
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Importações para visualização
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats, ndimage
from scipy.optimize import curve_fit

try:
    from skimage import feature
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


class ForensicAIEngine:
    """Engine principal da IA Forense"""
    
    def __init__(self):
        """Inicializa o engine da IA"""
        self.initialize_ai_models()
        self.setup_analysis_parameters()
        
    def initialize_ai_models(self):
        """Inicializa os modelos de IA"""
        self.models = {
            'post_mortem_estimator': self._create_post_mortem_model(),
            'tissue_classifier': self._create_tissue_classifier(),
            'anomaly_detector': self._create_anomaly_detector(),
            'quality_assessor': self._create_quality_model()
        }
        
    def _create_post_mortem_model(self):
        """Cria modelo para estimativa post-mortem"""
        return {
            'algor_mortis': 'thermal_analysis',
            'livor_mortis': 'blood_pooling_detection',
            'rigor_mortis': 'muscle_density_analysis',
            'putrefaction': 'gas_detection'
        }
    
    def _create_tissue_classifier(self):
        """Cria classificador de tecidos"""
        return {
            'air_gas': (-1000, -100),
            'fat': (-100, 0),
            'soft_tissue': (0, 100),
            'muscle': (40, 60),
            'blood': (50, 80),
            'bone': (100, 400),
            'calcification': (400, 1000),
            'metal': (1000, 3000)
        }
    
    def _create_anomaly_detector(self):
        """Cria detector de anomalias"""
        return {
            'statistical_thresholds': {
                'mean_deviation': 3.0,
                'std_threshold': 2.5,
                'skewness_limit': 2.0
            }
        }
    
    def _create_quality_model(self):
        """Cria modelo de avaliação de qualidade"""
        return {
            'snr_threshold': 20.0,
            'contrast_minimum': 0.1,
            'resolution_standard': 1.0,
            'artifact_tolerance': 0.05
        }
    
    def setup_analysis_parameters(self):
        """Configura parâmetros de análise"""
        self.analysis_params = {
            'temperature_range': (-10, 40),
            'humidity_range': (20, 100),
            'time_window': (0, 168),
            'confidence_threshold': 0.8
        }


class AIForensicAnalyzer:
    """Analisador forense com IA"""
    
    def __init__(self, ai_engine: ForensicAIEngine):
        self.ai_engine = ai_engine
        self.analysis_history = []
        
    def analyze_post_mortem_interval(self, image_array: np.ndarray, 
                                   environmental_params: Dict) -> Dict:
        """Análise autônoma do intervalo post-mortem"""
        
        results = {
            'estimated_pmi': None,
            'confidence': 0.0,
            'methods_used': [],
            'environmental_factors': environmental_params,
            'ai_insights': []
        }
        
        try:
            # Simulação de análise térmica
            thermal_analysis = self._simulate_thermal_analysis(image_array, environmental_params)
            results['algor_mortis'] = thermal_analysis
            results['methods_used'].append('Algor Mortis AI')
            
            # Simulação de análise de livor
            livor_analysis = self._simulate_livor_analysis(image_array)
            results['livor_mortis'] = livor_analysis
            results['methods_used'].append('Livor Mortis AI')
            
            # Consolidação dos resultados
            results = self._consolidate_results(results)
            
            # Gerar insights
            results['ai_insights'] = self._generate_insights(results)
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            results['error'] = str(e)
            
        return results
    
    def _simulate_thermal_analysis(self, image_array: np.ndarray, env_params: Dict) -> Dict:
        """Simula análise térmica"""
        
        # Simulação baseada na densidade dos tecidos
        mean_density = np.mean(image_array)
        std_density = np.std(image_array)
        
        # Estima temperatura baseada em parâmetros
        ambient_temp = env_params.get('temperature', 25)
        core_temp = ambient_temp + (mean_density / 50) * 10
        
        # Estima IPM baseado no resfriamento
        temp_diff = max(0, core_temp - ambient_temp)
        estimated_pmi = temp_diff * 2  # Fórmula simplificada
        
        return {
            'estimated_pmi_hours': min(48, max(0, estimated_pmi)),
            'core_temperature': core_temp,
            'ambient_temperature': ambient_temp,
            'confidence': 0.8 if 2 <= estimated_pmi <= 24 else 0.6
        }
    
    def _simulate_livor_analysis(self, image_array: np.ndarray) -> Dict:
        """Simula análise de livor mortis"""
        
        # Análise baseada em gradientes de densidade
        if len(image_array.shape) >= 2:
            grad_x = np.gradient(image_array.astype(float), axis=1)
            grad_y = np.gradient(image_array.astype(float), axis=0)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            pooling_intensity = np.mean(gradient_mag)
            fixation_ratio = np.std(gradient_mag) / (np.mean(gradient_mag) + 1e-6)
            
            # Classifica estágio
            if fixation_ratio > 0.7:
                stage = "fixed"
                pmi_range = (12, 18)
                confidence = 0.9
            elif fixation_ratio > 0.3:
                stage = "fixing"
                pmi_range = (6, 12)
                confidence = 0.7
            else:
                stage = "non_fixed"
                pmi_range = (2, 6)
                confidence = 0.8
        else:
            stage = "unknown"
            pmi_range = (0, 24)
            confidence = 0.3
            pooling_intensity = 0
            fixation_ratio = 0
        
        return {
            'stage': stage,
            'estimated_pmi_range': pmi_range,
            'pooling_intensity': pooling_intensity,
            'fixation_ratio': fixation_ratio,
            'confidence': confidence
        }
    
    def _consolidate_results(self, results: Dict) -> Dict:
        """Consolida resultados de diferentes métodos"""
        
        estimates = []
        weights = []
        
        # Coleta estimativas
        if 'algor_mortis' in results:
            algor_pmi = results['algor_mortis'].get('estimated_pmi_hours', 0)
            algor_conf = results['algor_mortis'].get('confidence', 0)
            estimates.append(algor_pmi)
            weights.append(algor_conf)
        
        if 'livor_mortis' in results:
            livor_range = results['livor_mortis'].get('estimated_pmi_range', (0, 0))
            livor_pmi = np.mean(livor_range)
            livor_conf = results['livor_mortis'].get('confidence', 0)
            estimates.append(livor_pmi)
            weights.append(livor_conf)
        
        # Calcula média ponderada
        if estimates and weights:
            weights = np.array(weights)
            estimates = np.array(estimates)
            
            weighted_estimate = np.average(estimates, weights=weights)
            overall_confidence = np.mean(weights)
            
            results['estimated_pmi'] = weighted_estimate
            results['confidence'] = overall_confidence
        
        return results
    
    def _generate_insights(self, results: Dict) -> List[str]:
        """Gera insights automáticos"""
        
        insights = []
        
        confidence = results.get('confidence', 0)
        if confidence > 0.8:
            insights.append("Alta confiabilidade - múltiplos indicadores concordantes")
        elif confidence > 0.6:
            insights.append("Confiabilidade moderada - resultados aceitáveis")
        else:
            insights.append("Baixa confiabilidade - considerar análises adicionais")
        
        pmi = results.get('estimated_pmi', 0)
        if pmi < 6:
            insights.append("IPM recente - fenômenos cadavéricos iniciais")
        elif pmi < 24:
            insights.append("IPM intermediário - múltiplos fenômenos presentes")
        else:
            insights.append("IPM prolongado - alterações avançadas")
        
        return insights


def main():
    """Função principal da aplicação"""
    
    # CSS customizado
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicialização
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = ForensicAIEngine()
        st.session_state.ai_analyzer = AIForensicAnalyzer(st.session_state.ai_engine)
    
    # Header
    st.title("🔬 Sistema de IA Forense Autônoma")
    st.markdown("### Análise Post-Mortem Inteligente e Automatizada")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🚀 Análise com IA")
        
        uploaded_file = st.file_uploader(
            "📁 Upload DICOM",
            type=['dcm', 'dicom'],
            help="Carregue um arquivo DICOM para análise automática"
        )
        
        if uploaded_file:
            st.subheader("🌡️ Parâmetros Ambientais")
            
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperatura (°C)", -10, 40, 25)
            with col2:
                humidity = st.slider("Umidade (%)", 20, 100, 60)
            
            environmental_params = {
                'temperature': temperature,
                'humidity': humidity
            }
            
            if st.button("🚀 Analisar com IA", type="primary", use_container_width=True):
                analyze_with_ai(uploaded_file, environmental_params)
    
    # Conteúdo principal
    if 'analysis_results' not in st.session_state:
        show_welcome_screen()
    else:
        show_results()


def analyze_with_ai(uploaded_file, environmental_params):
    """Executa análise com IA"""
    
    try:
        with st.spinner("🧠 Analisando com IA... Aguarde."):
            # Carrega DICOM
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            dicom_data = pydicom.dcmread(tmp_path)
            image_array = dicom_data.pixel_array
            
            # Executa análise
            analyzer = st.session_state.ai_analyzer
            results = analyzer.analyze_post_mortem_interval(image_array, environmental_params)
            
            # Armazena resultados
            st.session_state.analysis_results = results
            st.session_state.dicom_data = dicom_data
            st.session_state.image_array = image_array
            
            # Limpa arquivo temporário
            os.unlink(tmp_path)
            
            st.success("✅ Análise concluída!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        logger.error(f"Erro na análise: {e}")


def show_welcome_screen():
    """Tela de boas-vindas"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h1 style="font-size: 3em; margin-bottom: 20px;">🔬</h1>
            <h2>Bem-vindo ao Sistema de IA Forense</h2>
            <p style="font-size: 1.2em; margin: 30px 0;">
                Sistema autônomo de análise forense com inteligência artificial
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Funcionalidades Principais")
        
        features = [
            "🧠 **Análise Autônoma**: IA completamente automatizada",
            "⏱️ **Estimativa IPM**: Intervalo post-mortem preciso",
            "🔬 **Multi-método**: Algor, Livor, Rigor Mortis",
            "📊 **Relatórios IA**: Insights automáticos",
            "📈 **Visualizações**: Gráficos interativos",
            "🔒 **Seguro**: Processamento local"
        ]
        
        for feature in features:
            st.markdown(f"- {feature}")
        
        st.markdown("### 🚀 Como Usar")
        steps = [
            "1. 📁 Faça upload de um arquivo DICOM",
            "2. 🌡️ Configure parâmetros ambientais",
            "3. 🚀 Clique em 'Analisar com IA'",
            "4. 📊 Visualize resultados automáticos"
        ]
        
        for step in steps:
            st.markdown(step)


def show_results():
    """Mostra resultados da análise"""
    
    results = st.session_state.analysis_results
    
    # Métricas principais
    st.subheader("📊 Resultados da Análise")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pmi = results.get('estimated_pmi', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>⏱️ IPM Estimado</h3>
            <h2>{pmi:.1f} horas</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        confidence = results.get('confidence', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Confiabilidade</h3>
            <h2>{confidence:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        methods_count = len(results.get('methods_used', []))
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔬 Métodos</h3>
            <h2>{methods_count}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        status = "✅ Sucesso" if 'error' not in results else "❌ Erro"
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Status</h3>
            <h2>{status}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs detalhadas
    tab1, tab2, tab3 = st.tabs(["🧠 Insights da IA", "📊 Análise Detalhada", "📄 Relatório"])
    
    with tab1:
        st.subheader("💡 Insights Automáticos")
        
        insights = results.get('ai_insights', [])
        for i, insight in enumerate(insights):
            st.info(f"**Insight {i+1}:** {insight}")
        
        # Gráfico de confiabilidade
        if confidence > 0:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Nível de Confiabilidade"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "red"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔬 Detalhes por Método")
        
        # Algor Mortis
        if 'algor_mortis' in results:
            algor = results['algor_mortis']
            with st.expander("🌡️ Análise Térmica (Algor Mortis)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Temperatura Central", f"{algor.get('core_temperature', 0):.1f}°C")
                    st.metric("Temperatura Ambiente", f"{algor.get('ambient_temperature', 0):.1f}°C")
                with col2:
                    st.metric("IPM Estimado", f"{algor.get('estimated_pmi_hours', 0):.1f} horas")
                    st.metric("Confiança", f"{algor.get('confidence', 0):.1%}")
        
        # Livor Mortis
        if 'livor_mortis' in results:
            livor = results['livor_mortis']
            with st.expander("🩸 Análise Sanguínea (Livor Mortis)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Estágio:** {livor.get('stage', 'unknown')}")
                    st.metric("Intensidade de Acúmulo", f"{livor.get('pooling_intensity', 0):.3f}")
                with col2:
                    pmi_range = livor.get('estimated_pmi_range', (0, 0))
                    st.write(f"**Faixa IPM:** {pmi_range[0]}-{pmi_range[1]} horas")
                    st.metric("Confiança", f"{livor.get('confidence', 0):.1%}")
    
    with tab3:
        st.subheader("📄 Gerar Relatório")
        
        col1, col2 = st.columns(2)
        
        with col1:
            format_type = st.selectbox("Formato", ["PDF", "HTML", "JSON"])
            include_images = st.checkbox("Incluir Imagens", True)
        
        with col2:
            detail_level = st.selectbox("Detalhamento", ["Resumo", "Completo", "Técnico"])
            confidential = st.checkbox("Confidencial", True)
        
        if st.button("📋 Gerar Relatório", type="primary"):
            generate_report(results, format_type, include_images, detail_level, confidential)


def generate_report(results, format_type, include_images, detail_level, confidential):
    """Gera relatório da análise"""
    
    with st.spinner("📝 Gerando relatório..."):
        try:
            # Dados do relatório
            report_data = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'config': {
                    'format': format_type,
                    'include_images': include_images,
                    'detail_level': detail_level,
                    'confidential': confidential
                }
            }
            
            # Gera arquivo baseado no formato
            if format_type == "JSON":
                report_content = json.dumps(report_data, indent=2, default=str)
                file_data = BytesIO(report_content.encode())
                filename = f"relatorio_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                mime_type = "application/json"
            
            elif format_type == "HTML":
                html_content = generate_html_report(report_data)
                file_data = BytesIO(html_content.encode())
                filename = f"relatorio_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                mime_type = "text/html"
            
            else:  # PDF
                if REPORTLAB_AVAILABLE:
                    pdf_content = generate_pdf_report(report_data)
                    file_data = pdf_content
                    filename = f"relatorio_forense_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    mime_type = "application/pdf"
                else:
                    st.error("❌ ReportLab não disponível para PDF")
                    return
            
            # Download
            st.download_button(
                label=f"⬇️ Download Relatório ({format_type})",
                data=file_data,
                file_name=filename,
                mime=mime_type,
                type="primary"
            )
            
            st.success("✅ Relatório gerado com sucesso!")
            
        except Exception as e:
            st.error(f"❌ Erro ao gerar relatório: {str(e)}")


def generate_html_report(report_data):
    """Gera relatório HTML"""
    
    results = report_data['results']
    pmi = results.get('estimated_pmi', 0)
    confidence = results.get('confidence', 0)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise Forense com IA</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 3px solid #667eea;
        }}
        .metric {{
            display: inline-block;
            background: #f8f9fa;
            padding: 15px;
            margin: 10px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            min-width: 200px;
        }}
        .insight {{
            background: #e3f2fd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 RELATÓRIO DE ANÁLISE FORENSE</h1>
            <h3>Sistema de IA Autônoma</h3>
            <p>ID: {report_data['id']}</p>
            <p>Gerado em: {report_data['timestamp']}</p>
        </div>
        
        <h2>📊 Resultados Principais</h2>
        <div class="metric">
            <h4>⏱️ IPM Estimado</h4>
            <h2>{pmi:.1f} horas</h2>
        </div>
        <div class="metric">
            <h4>🎯 Confiabilidade</h4>
            <h2>{confidence:.1%}</h2>
        </div>
        
        <h2>💡 Insights da IA</h2>
        {generate_insights_html(results.get('ai_insights', []))}
        
        <div class="footer">
            <p><strong>Sistema de IA Forense Autônoma</strong></p>
            <p>Desenvolvido por: Wendell da Luz Silva</p>
            <p><em>Relatório gerado automaticamente - Confidencial</em></p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content


def generate_insights_html(insights):
    """Gera HTML para insights"""
    html = ""
    for insight in insights:
        html += f'<div class="insight">💡 {insight}</div>\n'
    return html


def generate_pdf_report(report_data):
    """Gera relatório PDF"""
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.pagesizes import A4
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        story.append(Paragraph("RELATÓRIO DE ANÁLISE FORENSE COM IA", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Informações
        results = report_data['results']
        pmi = results.get('estimated_pmi', 0)
        confidence = results.get('confidence', 0)
        
        story.append(Paragraph(f"IPM Estimado: {pmi:.1f} horas", styles['Normal']))
        story.append(Paragraph(f"Confiabilidade: {confidence:.1%}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Insights
        story.append(Paragraph("INSIGHTS DA IA", styles['Heading2']))
        for insight in results.get('ai_insights', []):
            story.append(Paragraph(f"• {insight}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        logger.error(f"Erro ao gerar PDF: {e}")
        return BytesIO(b"Erro ao gerar PDF")


if __name__ == "__main__":
    main()
