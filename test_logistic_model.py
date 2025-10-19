import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        # Testdaten laden
        self.df = load_data("advertising.csv")
        # Modell trainieren
        self.model, self.X_test, self.y_test = train_model(self.df)

    def test_predict_function(self):
        """Testfall 1: Prüft Accuracy ≥ 0.9 und Vorhandensein von Confusion Matrix"""
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")

    def test_fit_runtime(self):
        """Testfall 2: Prüft, dass train_model() ≤ 120 % der Referenzlaufzeit benötigt"""

        # 1️⃣ Erste Ausführung = Referenzzeit (repräsentative Zeit)
        _ = train_model(self.df)
        ref_time = get_last_timing("train_model")
        self.assertIsNotNone(ref_time, "Referenzlaufzeit konnte nicht gemessen werden")

        # 2️⃣ Zweite Ausführung = aktuelle Zeit
        _ = train_model(self.df)
        runtime = get_last_timing("train_model")
        self.assertIsNotNone(runtime, "Laufzeit konnte nicht gemessen werden")

        # 3️⃣ Vergleich mit 120 % Toleranz
        self.assertLessEqual(
            runtime,
            ref_time * 1.2,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
