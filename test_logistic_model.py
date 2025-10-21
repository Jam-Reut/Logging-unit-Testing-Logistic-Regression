import unittest
import time
from logistic_model import load_data, train_model, evaluate_model

class TestLogisticRegressionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Starte Unit-Tests: test_logistic_model.py ===")
        print("Setup initiales Training wird ausgeführt...\n")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print("Setup abgeschlossen.\n")

    # =======================================
    # TESTFALL 1 – Vorhersagefunktion (predict)
    # =======================================
    def test_predict_function(self):
        print("======================================================")
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("======================================================\n")

        acc = evaluate_model(self.model, self.X_test, self.y_test)
        print(f"Accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # =======================================
    # TESTFALL 2 – Laufzeit der Trainingsfunktion (fit)
    # =======================================
    def test_fit_runtime(self):
        print("======================================================")
        print("TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("======================================================\n")

        start_ref = time.perf_counter()
        train_model(self.df)
        ref_time = time.perf_counter() - start_ref
        print(f"Referenzlauf abgeschlossen in {ref_time:.4f} sec\n")

        start_new = time.perf_counter()
        train_model(self.df)
        runtime = time.perf_counter() - start_new
        print(f"Testlauf abgeschlossen in {runtime:.4f} sec\n")

        limit = ref_time * 1.2
        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            runtime, limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )
        print("Ergebnis: TESTFALL 2 PASSED\n")

    @classmethod
    def tearDownClass(cls):
        print("======================================================")
        print("TESTERGEBNISSE: ALLE TESTS ABGESCHLOSSEN")
        print("======================================================\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
