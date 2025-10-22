import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

class TestLogisticRegressionModel(unittest.TestCase):

    def setUp(self):
        #print("\n=== Starte Unit-Tests ===")
        print("Setup: Initiales Training wird ausgeführt...\n")
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test = train_model(self.df)
        print("Setup abgeschlossen.\n")

    # ------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("======================================================")
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("======================================================")

        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        print(f"Accuracy: {accuracy:.3f}")
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")

        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("======================================================")
        print("TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("======================================================")

        # Referenzlauf
        _ = train_model(self.df)
        ref_time = get_last_timing("train_model")

        # Testlauf
        _ = train_model(self.df)
        runtime = get_last_timing("train_model")

        limit = ref_time * 1.2  # 120 % Toleranz

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("Ergebnis: TESTFALL 2 PASSED\n")
		print("=== Laufzeit-Analyse ===")
        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
