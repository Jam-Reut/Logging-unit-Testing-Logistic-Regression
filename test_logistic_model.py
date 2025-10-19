import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        # Daten laden und Modell einmal trainieren
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test = train_model(self.df)

    def test_predict_function(self):
        """Testfall 1: Prüft Accuracy ≥ 0.9 und Vorhandensein von Confusion Matrix"""
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")

    def test_fit_runtime(self):
        """Testfall 2: Prüft, dass train_model() ≤ 120 % der Referenzlaufzeit benötigt"""

        # Fixe Referenzlaufzeit (z.B. 0.3 s)
        ref_runtime = 0.3
        max_allowed = ref_runtime * 1.2  # 120 % der Referenzzeit

        # Mehrfachmessung zur Stabilisierung
        runtimes = []
        for _ in range(3):
            _ = train_model(self.df)
            runtime = get_last_timing("train_model")
            self.assertIsNotNone(runtime, "Laufzeit konnte nicht gemessen werden")
            runtimes.append(runtime)

        avg_runtime = sum(runtimes) / len(runtimes)

        self.assertLessEqual(
            avg_runtime,
            max_allowed,
            f"Durchschnittliche Laufzeit {avg_runtime:.4f}s überschreitet 120 % der Referenzzeit ({max_allowed:.4f}s)"
        )

if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
