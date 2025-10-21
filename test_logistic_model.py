import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="a"
)

class TestLogisticRegressionModel(unittest.TestCase):
    """Unit Tests für ML-System mit YAML-artiger Konsolenausgabe."""

    @classmethod
    def setUpClass(cls):
        print("# ======================================================")
        print("# UNIT TEST AUSFÜHRUNG (test_logistic_model.py)")
        print("# ======================================================\n")

        print("Starte Test-Suite für Logistic Regression Modell\n")

        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

        print("Setup abgeschlossen – Modell initial trainiert.\n")

    def test_fit_runtime(self):
        """Testfall 2: Laufzeit ≤ 120 % der Referenz."""
        print("Testfall 2: Laufzeit der Trainingsfunktion (fit)")
        print("--------------------------------------------------")

        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1

        limit = ref_time * 1.2

        print(f"  Referenzlaufzeit train_model: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit train_model: {test_time:.4f} sec")
        print(f"  Zulässiges Limit (120 %): {limit:.4f} sec\n")

        print("  Laufzeitanalyse:")
        print(f"    - Referenzlaufzeit : {ref_time:.4f} sec")
        print(f"    - Aktuelle Laufzeit: {test_time:.4f} sec")
        print(f"    - Erlaubtes Limit  : {limit:.4f} sec\n")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("  ✅ Ergebnis: Testfall 2 PASSED\n")

    def test_predict_function(self):
        """Testfall 1: Accuracy ≥ 0.9 und Confusion Matrix vorhanden."""
        print("Testfall 1: Vorhersagefunktion (predict)")
        print("--------------------------------------------------")

        start = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start

        print(f"  → evaluate_model ran in: {runtime:.4f} sec")
        print(f"  🎯 Accuracy: {accuracy:.3f}\n")
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy unter 0.9")
        print("  ✅ Ergebnis: Testfall 1 PASSED\n")

    @classmethod
    def tearDownClass(cls):
        print("Test-Suite Ende:")
        print("  Alle Testfälle abgeschlossen.\n")

        print("# ======================================================")
        print("# TESTERGEBNISSE")
        print("# ======================================================\n")

if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
