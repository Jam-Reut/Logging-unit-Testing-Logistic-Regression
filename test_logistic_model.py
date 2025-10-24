import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing
import sys

class TestLogisticRegressionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ----------------------------------------------------------
        # Referenzlauf einmalig vor allen Tests
        # ----------------------------------------------------------
        print("\n" + "─" * 70, flush=True)
        print("[TEST 2 LOGGING: Referenzlauf (einmalig vor allen Tests)]", flush=True)
        print("─" * 70 + "\n", flush=True)

        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

        print(f"Referenzlauf abgeschlossen – Referenzzeit: {cls.REFERENCE_TIME:.4f} sec\n", flush=True)
        sys.stdout.flush()

        # Alternative: feste Referenzzeit
        # cls.REFERENCE_TIME = 0.3971

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("\n" + "=" * 70, flush=True)
        print("TESTFALL 1: predict(): Vorhersagefunktion", flush=True)
        print("=" * 70 + "\n", flush=True)
        print("[TEST 1 LOGGING: Vorhersageprüfung]\n", flush=True)

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("\nErgebnis: TESTFALL 1 PASSED\n", flush=True)

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("\n" + "=" * 70, flush=True)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion", flush=True)
        print("=" * 70 + "\n", flush=True)
        print("[TEST 2 LOGGING: aktueller Lauf (im Unittest)]\n", flush=True)

        df = load_data("advertising.csv")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
        limit = ref_time * 1.2

        print("Laufzeitanalyse, um die gemessenen Zeiten nachvollziehen zu können:", flush=True)
        print(f" - Referenzlaufzeit: {ref_time:.4f} sec", flush=True)
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec", flush=True)
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n", flush=True)

        if runtime <= limit:
            print("Laufzeit liegt innerhalb der Toleranz.\n", flush=True)
        else:
            print("❌ Laufzeit überschreitet das Limit!\n", flush=True)

        self.assertLessEqual(runtime, limit)
        print("Ergebnis: TESTFALL 2 PASSED\n", flush=True)


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===\n", flush=True)
    unittest.main(argv=[""], exit=False)
