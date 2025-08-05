import unittest
import re
from logistic_model import load_data, train_model, evaluate_model

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test = train_model(self.df)

    def test_predict_function(self):
        # Prüfen, ob evaluate_model korrekte Logs produziert und Accuracy >= 0.9 ist
        with self.assertLogs(level='INFO') as log_cm:
            accuracy = evaluate_model(self.model, self.X_test, self.y_test)

        # Logs mit Accuracy finden
        accuracy_logs = [msg for msg in log_cm.output if "Accuracy" in msg]
        self.assertTrue(len(accuracy_logs) > 0, "Keine Accuracy Logs gefunden")

        # Logs mit Confusion Matrix finden
        cm_logs = [msg for msg in log_cm.output if "Confusion Matrix" in msg]
        self.assertTrue(len(cm_logs) > 0, "Keine Confusion Matrix Logs gefunden")

        # Accuracy ist groß genug
        self.assertGreaterEqual(accuracy, 0.9)

    def test_fit_runtime(self):
        # Prüfe, ob train_model Timer-Log mit Laufzeit vorhanden ist und Laufzeit OK
        with self.assertLogs(level='INFO') as log_cm:
            train_model(self.df)

        timer_logs = [msg for msg in log_cm.output if re.search(r'train_model executed in \d+\.\d+ sec', msg)]
        self.assertTrue(timer_logs, "Timer Log fehlt")

        match = re.search(r'train_model executed in (\d+\.\d+) sec', timer_logs[0])
        self.assertIsNotNone(match, "Laufzeit konnte nicht extrahiert werden")
        runtime = float(match.group(1))

        ref_time = 0.5  # Referenzzeit an dein System anpassen

        self.assertLessEqual(runtime, ref_time * 1.2, f"Laufzeit {runtime}s überschreitet 120% von {ref_time}s")

if __name__ == "__main__":
     unittest.main(argv=[''], exit=False)





  
