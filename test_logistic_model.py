import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticModel(unittest.TestCase):
    """Automatisierte Tests für Logging, Timing und Modellverhalten."""

    @classmethod
    def setUpClass(cls):
        """Initialisierung: Daten laden & Modell trainieren."""
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

    def test_predict_function(self):
        """Testfall 1:
        Prüft, dass die Vorhersagefunktion (predict) eine hohe Accuracy liefert.
        """
        accuracy, cm = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        self.assertEqual(cm.shape, (2, 2), "Confusion Matrix hat falsche Dimensionen")

    def test_fit_runtime(self):
        """Testfall 2:
        Prüft, dass die Trainingsfunktion (fit) ≤ 120 % der Referenzlaufzeit benötigt.
        """
        # Referenzlaufzeit (repräsentativ)
        start = time.time()
        _ = train_model(self.df)
        ref_time = time.time() - start

        # Neue Laufzeit
        start = time.time()
        _ = train_model(self.df)
        runtime = time.time() - start

        # Prüfung mit 120%-Grenze
        self.assertLessEqual(
            runtime,
            ref_time * 1.2,
            f"Laufzeit {runtime:.4f}s überschreitet 120% der Referenzzeit ({ref_time:.4f}s)"
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
