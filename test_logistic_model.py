import unittest
import logging
import sys
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# Einheitliche Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)


class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Logging flushen, dann drucken
        sys.stdout.flush(); sys.stderr.flush()
        print("\n" + "─" * 70)
        print("[TEST 2 LOGGING: Referenzlauf (einmalig vor allen Tests)]")
        print("─" * 70 + "\n")
        sys.stdout.flush(); sys.stderr.flush()

        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

        sys.stdout.flush(); sys.stderr.flush()
        print(f"Referenzlauf abgeschlossen – Referenzzeit: {cls.REFERENCE_TIME:.4f} sec\n")
        sys.stdout.flush(); sys.stderr.flush()

    # ------------------------------------------------
    # TEST 1
    # ------------------------------------------------
    def test_1_predict_function(self):
        sys.stdout.flush(); sys.stderr.flush()
        print("\n" + "=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70)
        print("\n[TEST 1 LOGGING: Vorhersageprüfung]\n")
        sys.stdout.flush(); sys.stderr.flush()

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9)
        print("Ergebnis: TESTFALL 1 PASSED\n")
        sys.stdout.flush(); sys.stderr.flush()

    # ------------------------------------------------
    # TEST 2
    # ------------------------------------------------
    def test_2_train_runtime(self):
        sys.stdout.flush(); sys.stderr.flush()
        print("\n" + "=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70)
        print("\n[TEST 2 LOGGING: aktueller Lauf (im Unittest)]\n")
        sys.stdout.flush(); sys.stderr.flush()

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
        limit = ref_time * 1.2

        print("Laufzeitanalyse, um die gemessenen Zeiten nachvollziehen zu können:")
        print(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime <= limit:
            print("Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            print("❌ Laufzeit überschreitet das Limit!\n")

        self.assertLessEqual(runtime, limit)
        print("Ergebnis: TESTFALL 2 PASSED\n")
        sys.stdout.flush(); sys.stderr.flush()


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
