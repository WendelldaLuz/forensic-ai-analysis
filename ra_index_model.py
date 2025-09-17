# app.py
from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from ra_index_model import RAIndexModel

app = Flask(__name__)

# Carregar modelos
try:
    ra_model = joblib.load('models/ra_index_model.pkl')
except:
    ra_model = RAIndexModel()
    joblib.dump(ra_model, 'models/ra_index_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ra_index/data')
def get_ra_index_data():
    # Gerar dados de exemplo
    data = ra_model.generate_sample_data(n_samples=100)
    return jsonify(data.to_dict(orient='records'))

@app.route('/api/ra_index/analysis')
def get_ra_index_analysis():
    data = ra_model.generate_sample_data(n_samples=100)
    results = ra_model.train_and_evaluate(data)
    
    # Gerar visualização
    img = io.BytesIO()
    ra_model.visualize_results(data, results, save_path=img)
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    
    return jsonify({
        'correlation_with_pmi': results['correlation_with_pmi'],
        'visualization': f"data:image/png;base64,{img_base64}"
    })

@app.route('/api/model_comparison')
def get_model_comparison():
    # Simular dados de comparação com outros modelos
    models = ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting', 'RA-Index']
    performance = [0.85, 0.92, 0.78, 0.89, 0.95]  # AUC scores
    
    return jsonify({
        'models': models,
        'performance': performance
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)
