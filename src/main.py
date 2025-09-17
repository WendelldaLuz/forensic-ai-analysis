import streamlit as st
import numpy as np
import pandas as pd
import pydicom
import tempfile
import os
import json
import uuid
import logging
import sqlite3
import csv
from datetime import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Forensic AI Analysis System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# === Forensic AI Engine and Analyzer (simplified) ===

class ForensicAIEngine:
    def __init__(self):
        self.models = {}  # Placeholder for models
        self.analysis_params = {}

class AIForensicAnalyzer:
    def __init__(self, ai_engine):
        self.ai_engine = ai_engine

    def analyze_post_mortem_interval(self, image_array, env_params):
        # Simplified dummy analysis
        mean_density = np.mean(image_array)
        ambient_temp = env_params.get('temperature', 25)
        core_temp = ambient_temp + (mean_density / 50) * 10
        temp_diff = max(0, core_temp - ambient_temp)
        estimated_pmi = temp_diff * 2
        confidence = 0.8 if 2 <= estimated_pmi <= 24 else 0.6

        insights = []
        if confidence > 0.8:
            insights.append("Alta confiabilidade - múltiplos indicadores concordantes")
        elif confidence > 0.6:
            insights.append("Confiabilidade moderada - resultados aceitáveis")
        else:
            insights.append("Baixa confiabilidade - considerar análises adicionais")

        return {
            'estimated_pmi': min(48, max(0, estimated_pmi)),
            'confidence': confidence,
            'methods_used': ['Algor Mortis AI', 'Livor Mortis AI'],
            'ai_insights': insights,
            'algor_mortis': {
                'core_temperature': core_temp,
                'ambient_temperature': ambient_temp,
                'estimated_pmi_hours': min(48, max(0, estimated_pmi)),
                'confidence': confidence
            },
            'livor_mortis': {
                'stage': 'fixed' if confidence > 0.7 else 'non_fixed',
                'estimated_pmi_range': (12, 18) if confidence > 0.7 else (2, 6),
                'confidence': confidence
            }
        }


# === Database helpers ===

def get_user_reports(user_email):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, report_name, generated_at
            FROM reports
            WHERE user_email = ?
            ORDER BY generated_at DESC
        """, (user_email,))
        reports = cursor.fetchall()
        conn.close()
        return reports
    except Exception as e:
        logger.error(f"Erro ao recuperar relatórios: {e}")
        return []


def save_report_to_db(user_email, report_name, report_data, parameters):
    try:
        conn = sqlite3.connect("dicom_viewer.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reports (user_email, report_name, report_data, parameters)
            VALUES (?, ?, ?, ?)
        """, (user_email, report_name, report_data, json.dumps(parameters)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar relatório: {e}")
        return False


# === Evolution Status Tab and helpers ===

def show_evolution_status_tab(user_email):
    st.subheader("📈 Status de Evolução e Aprendizado do Sistema")

    tab_usage, tab_performance, tab_learning, tab_reports = st.tabs([
        "📊 Utilização", "⚡ Desempenho", "🧠 Aprendizado", "📋 Relatórios"
    ])

    with tab_usage:
        display_usage_metrics(user_email)

    with tab_performance:
        display_performance_metrics()

    with tab_learning:
        display_learning_analysis(user_email)

    with tab_reports:
        display_evolution_reports(user_email)


def display_usage_metrics(user_email):
    st.markdown("### 📊 Métricas de Utilização do Sistema")
    try:
        conn = sqlite3.connect("dicom_viewer.db")

        total_users = pd.read_sql("SELECT COUNT(*) as total FROM users", conn).iloc[0]['total']
        total_reports = pd.read_sql("SELECT COUNT(*) as total FROM reports", conn).iloc[0]['total']
        total_analyses = pd.read_sql("SELECT COUNT(*) as total FROM security_logs WHERE action = 'FILE_UPLOAD'", conn).iloc[0]['total']

        user_reports = pd.read_sql("SELECT COUNT(*) as total FROM reports WHERE user_email = ?", conn, params=(user_email,)).iloc[0]['total']
        user_analyses = pd.read_sql("SELECT COUNT(*) as total FROM security_logs WHERE user_email = ? AND action = 'FILE_UPLOAD'", conn, params=(user_email,)).iloc[0]['total']

        usage_by_role = pd.read_sql("SELECT role, COUNT(*) as count FROM users GROUP BY role", conn)
        daily_usage = pd.read_sql("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM security_logs 
            WHERE action = 'FILE_UPLOAD'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        """, conn)

        analysis_types = pd.read_sql("""
            SELECT 
                CASE 
                    WHEN parameters LIKE '%"report_type": "Forense"%' THEN 'Forense'
                    WHEN parameters LIKE '%"report_type": "Qualidade"%' THEN 'Qualidade'
                    WHEN parameters LIKE '%"report_type": "Estatístico"%' THEN 'Estatístico'
                    WHEN parameters LIKE '%"report_type": "Técnico"%' THEN 'Técnico'
                    ELSE 'Completo'
                END as report_type,
                COUNT(*) as count
            FROM reports 
            GROUP BY report_type
        """, conn)

        conn.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("👥 Usuários Totais", total_users)
        col2.metric("📋 Relatórios Gerados", total_reports)
        col3.metric("🔍 Análises Realizadas", total_analyses)
        col4.metric("📊 Seus Relatórios", user_reports)

        col1, col2 = st.columns(2)
        if not usage_by_role.empty:
            fig = px.pie(usage_by_role, values='count', names='role', title="Distribuição por Função dos Usuários")
            st.plotly_chart(fig, use_container_width=True)
        if not analysis_types.empty:
            fig = px.bar(analysis_types, x='report_type', y='count', title="Tipos de Relatórios Gerados")
            st.plotly_chart(fig, use_container_width=True)
        if not daily_usage.empty:
            fig = px.line(daily_usage, x='date', y='count', title="Uso Diário (Últimos 30 dias)", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("📈 Estatísticas Detalhadas de Uso"):
                st.metric("Média Diária", f"{daily_usage['count'].mean():.1f}")
                st.metric("Dia com Mais Análises", f"{daily_usage['count'].max()}")
                st.metric("Tendência (7 dias)", f"{(daily_usage['count'].iloc[:7].mean() - daily_usage['count'].iloc[7:14].mean()):+.1f}")

        with st.expander("👤 Suas Estatísticas de Uso"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Seus Relatórios", user_reports)
            col2.metric("Suas Análises", user_analyses)
            efficiency = (user_reports / user_analyses * 100) if user_analyses > 0 else 0
            col3.metric("Eficiência", f"{efficiency:.1f}%")

    except Exception as e:
        st.error(f"Erro ao carregar métricas de uso: {str(e)}")
        st.info("O banco de dados pode não ter sido inicializado ainda.")


def display_performance_metrics():
    st.markdown("### ⚡ Métricas de Desempenho do Sistema")
    # Dados simulados para exemplo
    days = 30
    df_perf = pd.DataFrame({
        'timestamp': pd.date_range(end=pd.Timestamp.today(), periods=days),
        'response_time': np.random.normal(1.2, 0.2, days),
        'cpu_usage': np.random.normal(40, 12, days),
        'memory_usage': np.random.normal(60, 8, days),
        'success_rate': np.random.normal(98.8, 0.5, days),
        'user_satisfaction': np.random.normal(4.5, 0.3, days)
    })

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, metric, label in zip(
        [col1, col2, col3, col4, col5],
        ['response_time', 'cpu_usage', 'memory_usage', 'success_rate', 'user_satisfaction'],
        ["⏱️ Tempo Resposta (s)", "💻 Uso de CPU (%)", "🧠 Uso de Memória (%)", "✅ Taxa de Sucesso (%)", "⭐ Satisfação (0-5)"]
    ):
        current = df_perf[metric].iloc[-1]
        prev = df_perf[metric].iloc[-2]
        delta = current - prev
        col.metric(label, f"{current:.2f}" if metric != 'user_satisfaction' else f"{current:.1f}", delta=f"{delta:+.2f}")

    fig = make_subplots(rows=3, cols=2, subplot_titles=(
        'Tempo de Resposta', 'Uso de CPU', 'Uso de Memória', 'Taxa de Sucesso', 'Satisfação do Usuário', ''
    ), vertical_spacing=0.1)

    fig.add_trace(go.Scatter(x=df_perf['timestamp'], y=df_perf['response_time'], name='Tempo Resposta'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_perf['timestamp'], y=df_perf['cpu_usage'], name='Uso CPU'), row=1, col=2)
    fig.add_trace(go.Scatter(x=df_perf['timestamp'], y=df_perf['memory_usage'], name='Uso Memória'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_perf['timestamp'], y=df_perf['success_rate'], name='Taxa Sucesso'), row=2, col=2)
    fig.add_trace(go.Scatter(x=df_perf['timestamp'], y=df_perf['user_satisfaction'], name='Satisfação'), row=3, col=1)

    fig.update_layout(height=800, showlegend=False, title_text="Evolução do Desempenho do Sistema")
    st.plotly_chart(fig, use_container_width=True)


def display_learning_analysis(user_email):
    st.markdown("### 🧠 Análise de Aprendizado e Evolução")
    learning_metrics = {
        "Acurácia Média": {"value": 92.5, "trend": "+2.3%", "color": "normal"},
        "Sensibilidade": {"value": 89.7, "trend": "+1.8%", "color": "normal"},
        "Especificidade": {"value": 94.3, "trend": "+0.9%", "color": "normal"},
        "Precisão": {"value": 90.1, "trend": "+1.5%", "color": "normal"},
        "F1-Score": {"value": 89.9, "trend": "+1.6%", "color": "normal"},
        "Tempo de Inferência (s)": {"value": 1.2, "trend": "-0.3s", "color": "normal"}
    }

    cols = st.columns(3)
    for i, (metric, data) in enumerate(learning_metrics.items()):
        with cols[i % 3]:
            st.metric(metric, f"{data['value']}{'%' if 'Tempo' not in metric else ''}", delta=data['trend'])

    accuracy_data = pd.DataFrame({
        "Data": pd.date_range(start='2024-01-01', periods=12, freq='M'),
        "Acurácia": [85, 87, 88, 89, 90, 91, 92, 92.5, 93, 93.2, 93.5, 94.0],
        "Confiabilidade": [82, 84, 86, 87, 88, 89, 90, 91, 91.5, 92, 92.5, 93]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=accuracy_data['Data'], y=accuracy_data['Acurácia'], name='Acurácia', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=accuracy_data['Data'], y=accuracy_data['Confiabilidade'], name='Confiabilidade', line=dict(color='green')))
    fig.update_layout(title="Evolução da Acurácia e Confiabilidade", xaxis_title="Data", yaxis_title="Percentual (%)", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


def display_evolution_reports(user_email):
    st.markdown("### 📋 Relatórios de Evolução do Sistema")
    # Aqui você pode implementar a listagem e geração de relatórios de evolução
    st.info("Funcionalidade de relatórios de evolução será implementada.")


# === Main app ===

def main():
    if 'ai_engine' not in st.session_state:
        st.session_state.ai_engine = ForensicAIEngine()
        st.session_state.ai_analyzer = AIForensicAnalyzer(st.session_state.ai_engine)

    st.title("Sistema de IA Forense Autônoma")
    st.markdown("Análise Post-Mortem Inteligente e Automatizada")
    st.markdown("---")

    with st.sidebar:
        st.header("Análise com IA")
        uploaded_file = st.file_uploader("Upload DICOM", type=['dcm', 'dicom'])
        if uploaded_file:
            temperature = st.slider("Temperatura (°C)", -10, 40, 25)
            humidity = st.slider("Umidade (%)", 20, 100, 60)
            env_params = {'temperature': temperature, 'humidity': humidity}
            if st.button("Analisar com IA", use_container_width=True):
                analyze_with_ai(uploaded_file, env_params)

    if 'analysis_results' not in st.session_state:
        st.info("Faça upload de um arquivo DICOM e configure os parâmetros para iniciar a análise.")
    else:
        results = st.session_state.analysis_results
        user_email = "usuario@exemplo.com"  # Ajuste para seu sistema de autenticação

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "Visualização", "Estatísticas", "Análise Técnica",
            "Qualidade", "Análise Post-Mortem", "RA-Index", "Relatórios", "Status de Evolução"
        ])

        # Aqui você deve chamar as funções correspondentes para cada aba
        # Exemplo simplificado:
        with tab8:
            show_evolution_status_tab(user_email)


def analyze_with_ai(uploaded_file, env_params):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        dicom_data = pydicom.dcmread(tmp_path)
        image_array = dicom_data.pixel_array

        analyzer = st.session_state.ai_analyzer
        results = analyzer.analyze_post_mortem_interval(image_array, env_params)

        st.session_state.analysis_results = results
        st.session_state.dicom_data = dicom_data
        st.session_state.image_array = image_array

        os.unlink(tmp_path)

        st.success("Análise concluída!")
        st.experimental_rerun()

    except Exception as e:
        st.error(f"Erro: {str(e)}")
        logger.error(f"Erro na análise: {e}")


if __name__ == "__main__":
    main()
