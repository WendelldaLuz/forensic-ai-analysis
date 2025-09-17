import json
import os
import datetime
from pathlib import Path

class LearningTracker:
    def __init__(self, log_file="logs/learning_evolution.json"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {
            "start_date": datetime.datetime.now().isoformat(),
            "total_analyses": 0,
            "correct_predictions": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "accuracy_history": [],
            "model_versions": [],
            "performance_metrics": []
        }
    
    def save_history(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def record_analysis(self, was_correct, is_false_positive=False, is_false_negative=False):
        self.history["total_analyses"] += 1
        
        if was_correct:
            self.history["correct_predictions"] += 1
        
        if is_false_positive:
            self.history["false_positives"] += 1
        
        if is_false_negative:
            self.history["false_negatives"] += 1
        
        # Calcular acurácia atual
        if self.history["total_analyses"] > 0:
            accuracy = (self.history["correct_predictions"] / self.history["total_analyses"]) * 100
            self.history["accuracy_history"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "accuracy": round(accuracy, 2),
                "total_analyses": self.history["total_analyses"]
            })
        
        self.save_history()
    
    def record_model_update(self, version, improvements):
        self.history["model_versions"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "version": version,
            "improvements": improvements,
            "metrics_before": self.get_current_metrics()
        })
    
    def get_current_metrics(self):
        total = self.history["total_analyses"]
        correct = self.history["correct_predictions"]
        
        metrics = {
            "total_analyses": total,
            "correct_predictions": correct,
            "false_positives": self.history["false_positives"],
            "false_negatives": self.history["false_negatives"],
            "accuracy": round((correct / total * 100), 2) if total > 0 else 0,
            "precision": self.calculate_precision(),
            "recall": self.calculate_recall()
        }
        return metrics
    
    def calculate_precision(self):
        tp = self.history["correct_predictions"]
        fp = self.history["false_positives"]
        return round((tp / (tp + fp)) * 100, 2) if (tp + fp) > 0 else 0
    
    def calculate_recall(self):
        tp = self.history["correct_predictions"]
        fn = self.history["false_negatives"]
        return round((tp / (tp + fn)) * 100, 2) if (tp + fn) > 0 else 0
    
    def get_learning_progress(self):
        history = self.history["accuracy_history"]
        if len(history) < 2:
            return "Dados insuficientes para calcular progresso"
        
        initial_acc = history[0]["accuracy"]
        current_acc = history[-1]["accuracy"]
        improvement = current_acc - initial_acc
        
        return {
            "initial_accuracy": initial_acc,
            "current_accuracy": current_acc,
            "improvement": round(improvement, 2),
            "improvement_percentage": round((improvement / initial_acc * 100), 2) if initial_acc > 0 else 0,
            "days_operating": (datetime.datetime.now() - 
                             datetime.datetime.fromisoformat(self.history["start_date"])).days
        }
    
    def generate_progress_report(self):
        progress = self.get_learning_progress()
        metrics = self.get_current_metrics()
        
        report = {
            "relatorio_aprendizado": {
                "data_geracao": datetime.datetime.now().isoformat(),
                "periodo_operacao": f"{progress['days_operating']} dias",
                "metricas_atuais": metrics,
                "progresso_evolucao": progress,
                "historico_versoes": len(self.history["model_versions"]),
                "ultimas_10_analises": self.history["accuracy_history"][-10:] if len(self.history["accuracy_history"]) > 10 else self.history["accuracy_history"]
            }
        }
        
        return report

# Exemplo de uso integrado
if __name__ == "__main__":
    tracker = LearningTracker()
    
    # Simular algumas análises
    tracker.record_analysis(was_correct=True)
    tracker.record_analysis(was_correct=True)
    tracker.record_analysis(was_correct=False, is_false_positive=True)
    tracker.record_analysis(was_correct=True)
    
    # Registrar atualização do modelo
    tracker.record_model_update("1.2.0", ["Melhoria na detecção de PDF maliciosos", "Otimização de performance"])
    
    # Gerar relatório de progresso
    progress_report = tracker.generate_progress_report()
    print("Relatório de Evolução da IA:")
    print(json.dumps(progress_report, indent=2, ensure_ascii=False))
