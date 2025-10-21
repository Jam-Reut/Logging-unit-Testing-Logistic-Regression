"""
test_logistic_model.py
----------------------
Automatisierte Unittests für Logistic Regression Modell.
Testfall 1: Vorhersage (predict)
Testfall 2: Laufzeitanalyse (fit)
"""

import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("=== Starte Unit-Tests ===")
        print(" Setup initial trainieren\n")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print("  Setup abgeschlossen\n")

    # --------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # --------------------------------------------------
    def test_1_predict_function(self):
        print("=== Testfall 1: Vorhersagefunktion (predict) ===\n")

        start = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start

        print(f"  → evaluate_model ran in: {runtime:.4f} sec\n")
        print(f"  Accuracy: {accuracy:.3f}\n")
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy unter 0.9")
        print("Ergebnis: Testfall 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # --------------------------------------------------
    def test_2_fit_runtime(self):
        print("=== Testfall 2: Laufzeit der Trainingsfunktion (fit) ===\n")

        print("=== Laufzeitmessung: train_model ===\n")

        # 1️⃣ Referenzlaufzeit (Baseline)
        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0
        print(f"  Referenzlauf (Baseline) abgeschlossen in {ref_time:.4f} sec\n")

        # 2️⃣ Testlaufzeit
        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1
        print(f"  Testlauf abgeschlossen in {test_time:.4f} sec\n")

        limit = ref_time * 1.2

        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit (erster Trainingslauf): {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit (zweiter Trainingslauf): {test_time:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

    @classmethod
    def tearDownClass(cls):
        print("# ======================================================")
        print("# TESTERGEBNISSE")
        print("# ======================================================\n")
        print("Alle Testfälle abgeschlossen.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
