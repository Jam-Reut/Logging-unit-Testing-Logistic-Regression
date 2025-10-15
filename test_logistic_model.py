import unittest
import re
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticModel(unittest.TestCase):
    """
    Unit-Tests für das Logistic Regression Projekt.
    Testet:
    1) Vorhersagequalität über evaluate_model()
    2) Laufzeitüberwachung über train_model()
    """

    def setUp(self):
        """Vorbereitung: Daten laden und Modell trainieren"""
        self.df = load_data("advertising.csv")
        # train_model soll mindestens Modell, X_test, y_test zurückgeben
        result = train_model(self.df)
        if len(result) == 3:
            self.model, self.X_test, self.y_test = result
        else:
            # falls train_model z. B. 5 Werte liefert
            self.model, _, self.X_test, _, self.y_test = result

    # ------------------------------------------------------------------
    def test_predict_function(self):
        """Testfall 1: evaluate_model() liefert Accuracy ≥ 0.9 und loggt korrekt"""
        with self.assertLogs(level='INFO') as log_cm:
            accuracy = evaluate_model(self.model, self.X_test, self.y_test)

        # Prüfe, dass evaluate_model() einen Wert zurückgibt
        self.assertIsNotNone(accuracy, "evaluate_model() hat keinen Rückgabewert geliefert")

        # Prüfe Logeinträge auf Accuracy
        accuracy_logs = [msg for msg in log_cm.output if "Accuracy" in msg]
        self.assertTrue(accuracy_logs, "Keine Accuracy-Logs gefunden")

        # Prüfe Logeinträge auf Confusion Matrix
        cm_logs = [msg for msg in log_cm.output if "Confusion Matrix" in msg]
        self.assertTrue(cm_logs, "Keine Confusion-Matrix-Logs gefunden")

        # Prüfe Mindestgenauigkeit
        self.assertGreaterEqual(accuracy, 0.9, f"Accuracy zu niedrig: {accuracy:.2f}")

    # ------------------------------------------------------------------
    def test_fit_runtime(self):
        """Testfall 2: train_model() überschreitet 120 % der Referenzlaufzeit nicht"""

        # Schritt 1: Referenzzeit messen
        start = time.perf_counter()
        train_model(self.df)
        ref_time = time.perf_counter() - start

        # Schritt 2: Zweiten Lauf mit Logging überwachen
        with self.assertLogs(level='INFO') as log_cm:
            train_model(self.df)

        # Logzeilen finden, die die Laufzeit enthalten
        timer_logs = [msg for msg in log_cm.output if "train_model executed in" in msg]
        self.assertTrue(timer_logs, "Timer-Log fehlt")

        # Laufzeitwert aus dem Log extrahieren (robustes Regex)
        match = re.search(r"train_model executed in ([0-9]*\.?[0-9]+) sec", timer_logs[0])
        self.assertIsNotNone(match, "Laufzeit konnte nicht extrahiert werden")
        runtime = float(match.group(1))

        # Laufzeit darf max. 120 % der Referenzzeit betragen
        self.assertLessEqual(
            runtime, ref_time * 1.2,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % von {ref_time:.4f}s"
        )


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
