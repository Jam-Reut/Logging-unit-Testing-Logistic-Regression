import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ------------------------------------------------------------
# Logger-Konfiguration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)
plain = logging.getLogger("plain")
plain_handler = logging.StreamHandler()
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain.addHandler(plain_handler)
plain.propagate = False


# ------------------------------------------------------------
# Unit-Tests
# ------------------------------------------------------------
class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # einmalige repräsentative Referenzzeit
        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        plain.info("\n=== Starte Unit-Tests ===\n")
        plain.info("=" * 70)
        plain.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9)
        plain.info("Ergebnis: TESTFALL 1 PASSED ✅\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        plain.info("=" * 70)
        plain.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        plain.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
        limit = ref_time * 1.2

        plain.info("[TEST 2 LOGGING: aktuelle Laufzeit]\n")
        plain.info("Laufzeitanalyse:")
        plain.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        try:
            self.assertLessEqual(runtime, limit)
            plain.info("Laufzeit liegt innerhalb der Toleranz.\n")
            plain.info("Ergebnis: TESTFALL 2 PASSED ✅\n")
        except AssertionError:
            plain.info("❌ Laufzeit überschreitet das Limit!\n")
            plain.info("Ergebnis:
