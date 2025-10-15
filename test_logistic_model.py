import unittest
from logistic_model import load_data, train_model, evaluate_model

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        # Daten laden und ein Modell trainieren (für den predict-Test)
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test, _ = train_model(self.df)

    def test_predict_function(self):
        """Testfall 1: Prüft Accuracy und Confusion Matrix der Vorhersagefunktion"""
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy < 0.9")

    def test_fit_runtime(self):
        """Testfall 2: Prüft, dass train_model() <= 120 % der festen Referenzlaufzeit benötigt"""

        # Feste Referenzzeit (z. B. gemessen auf stabilem System vorher)
        ref_time = 0.3  # Sekunden

        # Aktuelle Laufzeit messen
        _, _, _, runtime = train_model(self.df)

        # 120 %-Grenze berechnen
        max_allowed = ref_time * 1.2

        # Prüfung mit klarer Fehlermeldung
        self.assertLessEqual(
            runtime,
            max_allowed,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit "
            f"({ref_time:.4f}s → Grenzwert {max_allowed:.4f}s)"
        )

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
