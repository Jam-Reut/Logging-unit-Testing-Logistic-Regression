import unittest
import pandas as pd
import logistic_model as lm
import logging
import io
from contextlib import redirect_stdout

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        self.df = lm.load_data("advertising.csv")

    def test_predict_function(self):
        """Test für predict(): Accuracy + Confusion Matrix + Logging"""
        with self.assertLogs(level='INFO') as log_cm:
            model, X_test, y_test = lm.train_model(self.df)
            acc, cm = lm.evaluate_model(model, X_test, y_test)

        # Prüfen ob Accuracy im Log ist
        log_messages = "\n".join(log_cm.output)
        self.assertIn("Accuracy:", log_messages)
        self.assertIn("Confusion Matrix", log_messages)
        self.assertGreaterEqual(acc, 0)  # Accuracy >= 0
        self.assertLessEqual(acc, 1)     # Accuracy <= 1

    def test_fit_runtime(self):
        """Test für fit() Laufzeit, max 120% eines Referenzwertes"""
        # Referenzlaufzeit messen
        f = io.StringIO()
        with redirect_stdout(f):
            model, X_test, y_test = lm.train_model(self.df)
        logs = f.getvalue()

        # Referenzzeit aus Log extrahieren
        ref_time = None
        for line in logs.splitlines():
            if "executed in" in line:
                try:
                    ref_time = float(line.split("executed in")[1].split("sec")[0])
                except:
                    pass

        self.assertIsNotNone(ref_time, "Timer Log fehlt")

        # Testlauf: prüfen, ob Zeit <= 120% der Referenz
        f = io.StringIO()
        with redirect_stdout(f):
            model, X_test, y_test = lm.train_model(self.df)
        logs_test = f.getvalue()

        test_time = None
        for line in logs_test.splitlines():
            if "executed in" in line:
                try:
                    test_time = float(line.split("executed in")[1].split("sec")[0])
                except:
                    pass

        self.assertIsNotNone(test_time, "Timer Log fehlt im Testlauf")
        self.assertLessEqual(test_time, ref_time * 1.2, "Trainingszeit überschreitet Grenzwert")

if __name__ == "__main__":
    unittest.main()
