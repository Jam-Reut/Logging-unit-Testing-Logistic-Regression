import unittest
import logging
import time
import pandas as pd
import logistic_model as lm
from sklearn.metrics import accuracy_score, confusion_matrix

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        # Logging-Level für Tests auf INFO, damit assertLogs funktioniert
        logging.getLogger().setLevel(logging.INFO)
        self.df = lm.load_data("advertising.csv")

    def test_predict_function(self):
        """Test für predict() inkl. Accuracy und Confusion Matrix + Logging prüfen"""
        model, X_test, y_test = lm.train_model(self.df)

        with self.assertLogs(level='INFO') as log_cm:
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            cm = confusion_matrix(y_test, predictions)

        # Prüfen, ob Logging Eintrag für predict vorhanden ist (du kannst 'Calling predict' o.Ä. erweitern)
        log_messages = "\n".join(log_cm.output)
        self.assertTrue(any("predict" in msg for msg in log_messages.lower()), "Predict Log fehlt")

        # Check Accuracy > 0.9 (angepasst an dein Modell)
        self.assertGreater(acc, 0.9)

        # Optional: Check Confusion Matrix Struktur (4 Felder)
        self.assertEqual(cm.size, 4)

    def test_fit_runtime(self):
        """Test für fit() Laufzeit, max 120% eines Referenzwertes"""

        # Referenzlaufzeit messen (z.B. beim ersten Durchlauf, hier als Beispiel hardcoded)
        start = time.perf_counter()
        model, _, _ = lm.train_model(self.df)
        end = time.perf_counter()
        reference_runtime = end - start

        max_allowed = reference_runtime * 1.2  # 120% Grenze

        # Laufzeit erneut messen und testen
        with self.assertLogs(level='INFO') as log_cm:
            start_test = time.perf_counter()
            model, _, _ = lm.train_model(self.df)
            end_test = time.perf_counter()

        runtime = end_test - start_test
        self.assertLessEqual(runtime, max_allowed, f"Laufzeit {runtime:.2f}s überschreitet max. {max_allowed:.2f}s")

        # Optional: Check ob Timer-Log im Output ist
        log_messages = "\n".join(log_cm.output)
        self.assertTrue(any("executed in" in msg for msg in log_messages), "Timer Log fehlt")

if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===\n")
    unittest.main()
