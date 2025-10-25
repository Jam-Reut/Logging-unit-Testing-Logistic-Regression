import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

logger = logging.getLogger(__name__)

class TestLogisticRegressionModel(unittest.TestCase):
    """Tests nach Ori Cohen: Logging + Timing + klare Outputstruktur"""

    @classmethod
    def setUpClass(cls):
        print("=== Starte Unit-Tests ===\n")
        df = load_data("advertising.csv")
        train_model(df)
        cls.ref_time = get_last_timing("train_model")

    def test_1_predict(self):
        print("=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9)
        print("\nErgebnis: TESTFALL 1 PASSED ✅\n")

    def test_2_train_runtime(self):
        print("=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")
        ref = self.ref_time
        limit = ref * 1.2

        print("Laufzeitanalyse:")
        print("  (Referenzzeit = aus setUpClass())")
        print(f" - Referenzlaufzeit: {ref:.4f} sec")
        print("  (Aktuelle Laufzeit = aktueller Testlauf)")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime <= limit:
            print("Laufzeit liegt innerhalb der Toleranz.\n")
            print("Ergebnis: TESTFALL 2 PASSED ✅\n")
        else:
            print("❌ Laufzeit überschreitet das Limit!\n")
            print("Ergebnis: TESTFALL 2 FAILED ❌\n")
            self.fail("Trainingslaufzeit überschreitet das zulässige Limit.")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
