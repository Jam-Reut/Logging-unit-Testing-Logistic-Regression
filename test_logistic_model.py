"""
test_logistic_model.py
----------------------
Automatisierte Unittests für das Logistic-Regression-Modell.
Beinhaltet:
- Testfall 1: Vorhersagefunktion (predict)
- Testfall 2: Laufzeitanalyse (fit)
"""

import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticRegressionModel(unittest.TestCase):
    """Testklasse für ML-System mit Logging und Zeitmessung."""

    # --------------------------------------------------
    # SETUP: einmalige Initialisierung
    # --------------------------------------------------
    @classmethod
    def setUpClass(cls):
        print("Setup initial trainieren\n")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

        print("\nSetup abgeschlossen\n")

    # --------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # --------------------------------------------------
    def test_1_predict_function(self):
        """Testfall 1: Accuracy ≥ 0.9 und Confusion Matrix vorhanden."""
        print("=== Testfall 1: Vorhersagefunktion (predict) ===\n")
        start = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start
        print(f"  → evaluate_model ran in: {runtime:.4f} sec")
        print(f"  Accuracy: {accuracy:.3f}\n")

        self.assertGreaterEqual(accuracy, 0.9, "Accuracy unter 0.9 – Modellvorhersage unzureichend.")
        print("Ergebnis: Testfall 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # --------------------------------------------------
    def test_2_fit_runtime(self):
        """Testfall 2: Überprüft, ob Laufzeit ≤ 120 % der Referenzzeit."""
        print("=== Testfall 2: Laufzeit der Trainingsfunktion (fit) ===\n")

        # 1️⃣ Referenzlaufzeit (erster Trainingslauf)
        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0

        # 2️⃣ Testlaufzeit (zweiter Trainingslauf)
        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1

        limit = ref_time * 1.2

        # Laufzeitanalyse ausgeben
        print("\n=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit (erster Trainingslauf): {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit (zweiter Trainingslauf): {test_time:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

    # --------------------------------------------------
    # TEARDOWN: Abschlussausgabe
    # --------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        print("# ======================================================")
        print("# TESTERGEBNISSE")
        print("# ======================================================\n")
        print("Alle Testfälle abgeschlossen.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
