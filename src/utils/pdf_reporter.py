import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

class PDFReporter:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_forensic_report(self, analysis_data, filename=None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forensic_report_{timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Criar documento PDF
        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Título
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("RELATÓRIO DE ANÁLISE FORENSE - IA", title_style))
        
        # Metadados
        story.append(Paragraph(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Resumo da análise
        story.append(Paragraph("RESUMO DA ANÁLISE", styles['Heading2']))
        story.append(Paragraph(f"Total de arquivos analisados: {analysis_data.get('total_files', 0)}", styles['Normal']))
        story.append(Paragraph(f"Arquivos suspeitos: {analysis_data.get('suspicious_files', 0)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Detalhes da análise (exemplo em tabela)
        if 'file_details' in analysis_data:
            story.append(Paragraph("DETALHES DOS ARQUIVOS", styles['Heading2']))
            
            data = [['Arquivo', 'Tipo', 'Status', 'Pontuação']]
            for file in analysis_data['file_details']:
                data.append([
                    file.get('name', 'N/A'),
                    file.get('type', 'N/A'),
                    file.get('status', 'N/A'),
                    str(file.get('score', 0))
                ])
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        # Estatísticas de aprendizado
        if 'learning_stats' in analysis_data:
            story.append(Spacer(1, 12))
            story.append(Paragraph("ESTATÍSTICAS DE APRENDIZADO DA IA", styles['Heading2']))
            
            for key, value in analysis_data['learning_stats'].items():
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        
        # Gerar PDF
        doc.build(story)
        return filepath

# Exemplo de uso:
if __name__ == "__main__":
    reporter = PDFReporter()
    
    sample_data = {
        'total_files': 15,
        'suspicious_files': 3,
        'file_details': [
            {'name': 'arquivo1.exe', 'type': 'executável', 'status': 'suspeito', 'score': 85},
            {'name': 'documento.pdf', 'type': 'documento', 'status': 'limpo', 'score': 10},
            {'name': 'script.php', 'type': 'script', 'status': 'malicioso', 'score': 95}
        ],
        'learning_stats': {
            'Total de análises': 150,
            'Taxa de acerto': '92%',
            'Modelo versão': '2.1.4'
        }
    }
    
    report_path = reporter.generate_forensic_report(sample_data)
    print(f"Relatório gerado: {report_path}")
