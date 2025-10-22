import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):
    # ------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("=" * 54)
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("=" * 54)
        print("Starte Testfall 1 – Validierung der Modellvorhersage...\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)

        accuracy = evaluate_model(model, X_test, y_test)
        print(f"\n→ evaluate_model ran in: {get_last_timing('evaluate_model'):.4f} sec")
        print(f"Accuracy: {accuracy:.3f}")

        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("=" * 54)
        print("TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("=" * 54)
        print("Starte Testfall 2 – Analyse der Trainingslaufzeit...\n")

        df = load_data("advertising.csv")

        # Referenzlauf
        print("=== Modelltraining (Referenzlauf) ===")
        train_model(df)
        ref_time = get_last_timing("train_model")
        print(f"→ train_model (Referenzlauf) ran in: {ref_time:.4f} sec\n")

        # Testlauf
        print("=== Modelltraining (Testlauf) ===")
        train_model(df)
        runtime = get_last_timing("train_model")
        print(f"→ train_model (Testlauf) ran in: {runtime:.4f} sec\n")

        limit = ref_time * 1.2

        # Ergebnisprüfung
        if runtime <= limit:
            result = "PASSED"
        else:
            result = "FAILED ❌"

        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")
        print(f"Ergebnis: TESTFALL 2 {result}\n")

        # Testbedingung ausführen
        self.assertLessEqual(runtime, limit,
                             f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)")


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
