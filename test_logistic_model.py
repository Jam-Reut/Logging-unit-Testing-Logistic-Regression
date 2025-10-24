import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ================================================================
# LOGGING gemäß Ori Cohen – Klar getrennt: Testausgabe vs. Technik
# ================================================================

# Technisches Logging (bereits in logistic_model.py eingerichtet)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)

# Separater Klartext-Logger (Testausgabe)
plain_logger = logging.getLogger("plain")
plain_logger.handlers.clear()
plain_handler = logging.StreamHandler()
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(plain_handler)
plain_logger.propagate = False


# ================================================================
# UNIT-TEST-KLASSE
# ================================================================
class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Einmaliger Referenzlauf zur Laufzeitmessung
        plain_logger.info("=== Starte Unit-Tests ===\n")
        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

    # ------------------------------------------------------------
    # TESTFALL 1 – Vorhersagefunktion
    # ------------------------------------------------------------
    def test_1_predict_function(self):
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain_logger.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc, metrics_text = evaluate_model(model, X_test, y_test)

        plain_logger.info(metrics_text)
        self.assertGreaterEqual(acc, 0.9)
        plain_logger.info("Ergebnis: TESTFALL 1 PASSED ✅\n")

    # ------------------------------------------------------------
    # TESTFALL 2 – Laufzeitprüfung
    # ------------------------------------------------------------
    def test_2_train_runtime(self):
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        plain_logger.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
        limit = ref_time * 1.2

        plain_logger.info("[TEST 2 LOGGING: aktuelle Laufzeit]\n")
        plain_logger.info("Laufzeitanalyse:")
        plain_logger.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain_logger.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain_logger.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        try:
            self.assertLessEqual(runtime, limit)
            plain_logger.info("Laufzeit liegt innerhalb der Toleranz.\n")
            plain_logger.info("Ergebnis: TESTFALL 2 PASSED ✅\n")
        except AssertionError:
            plain_logger.info("❌ Laufzeit überschreitet das Limit!\n")
            plain_logger.info("Ergebnis: TESTFALL 2 FAILED ❌\n")
            raise


# ================================================================
# TESTSTARTER
# ================================================================
if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
