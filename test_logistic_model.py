import unittest
import re
from logistic_model import load_data, train_model, evaluate_model

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test = train_model(self.df)

    def test_predict_function(self):
        """Testfall 1: Prüft Accuracy und Confusion Matrix der Vorhersagefunktion"""
        with self.assertLogs(level='INFO') as log_cm:
            accuracy = evaluate_model(self.model, self.X_test, self.y_test)

        accuracy_logs = [msg for msg in log_cm.output if "Accuracy" in msg]
        self.assertTrue(accuracy_logs, "Keine Accuracy Logs gefunden")

        cm_logs = [msg for msg in log_cm.output if "Confusion Matrix" in msg]
        self.assertTrue(cm_logs, "Keine Confusion Matrix Logs gefunden")

        self.assertGreaterEqual(accuracy, 0.9, "Accuracy < 0.9")

    def test_fit_runtime(self):
        """Testfall 2: Prüft, dass train_model() <= 120 % der Referenzlaufzeit benötigt"""
        
        # 1️⃣ Erste Messung – Referenzlaufzeit bestimmen
        with self.assertLogs(level='INFO') as log_cm1:
            train_model(self.df)
        ref_match = re.search(r'train_model executed in (\d+\.\d+) sec', "".join(log_cm1.output))
        self.assertIsNotNone(ref_match, "Referenzlaufzeit konnte nicht extrahiert werden")
        ref_time = float(ref_match.group(1))
      
        # 2️⃣ Zweite Messung – aktuelle Laufzeit prüfen
        with self.assertLogs(level='INFO') as log_cm2:
            train_model(self.df)
        run_match = re.search(r'train_model executed in (\d+\.\d+) sec', "".join(log_cm2.output))
        self.assertIsNotNone(run_match, "Laufzeit konnte nicht extrahiert werden")
        runtime = float(run_match.group(1))

        # 3️⃣ Vergleich mit 120 %-Grenze
        self.assertLessEqual(
            runtime,
            ref_time * 1.2,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
