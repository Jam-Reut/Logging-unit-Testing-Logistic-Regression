"""
test_logistic_model.py
----------------------
Automatisierte Unit-Tests für das Logistic Regression Modell.
Basierend auf dem Ansatz von Ori Cohen:
- Testfall 1: Vorhersage (Accuracy & Confusion Matrix)
- Testfall 2: Laufzeitanalyse (fit ≤ 120 % der Baseline)
"""

import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n=== Starte Unit-Tests: test_logistic_model.py ===")
        print("Setup initiales Training wird ausgeführt...\n")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print("Setup abgeschlossen.\n")

    # --------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # --------------------------------------------------
    def test_1_predict(self):
        print("======================================================")
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("======================================================")

        start = time.perf_counter()
        acc = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start

        print(f"TESTFALL 1 → evaluate_model ran in: {runtime:.4f} sec")
        print(f"Accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.9, "Accuracy zu niedrig (<0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # --------------------------------------------------
    def test_2_fit_runtime(self):
        print("======================================================")
        print("TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("======================================================")

        # Referenzlauf (Baseline)
        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0
        print(f"Referenzlauf abgeschlossen in {ref_time:.4f} sec")

        # Testlauf
        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1
        print(f"Testlauf abgeschlossen in {test_time:.4f} sec")

        limit = ref_time * 1.2
        print("\n=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {test_time:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )
        print("Ergebnis: TESTFALL 2 PASSED\n")

    @classmethod
    def tearDownClass(cls):
        print("======================================================")
        print("TESTERGEBNISSE: ALLE TESTS ABGESCHLOSSEN")
        print("======================================================\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
