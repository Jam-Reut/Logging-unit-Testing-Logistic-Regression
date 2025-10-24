import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ------------------------------------------------------------
# Logger-Konfiguration
# ------------------------------------------------------------
# Verhindert doppelte Handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# Nur EIN Plain-Logger ohne Verdopplung
plain = logging.getLogger("plain")
plain.handlers.clear()
plain_handler = logging.StreamHandler()
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain.addHandler(plain_handler)
plain.propagate = False


# ------------------------------------------------------------
# Unit-Testklasse
# ------------------------------------------------------------
class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --------------------------------------------------------
        # EINMALIGER REFERENZLAUF (vor allen Tests)
        # --------------------------------------------------------
        plain.info("\n" + "─" * 70)
        plain.info("[SETUP-LOGGING: Referenzlauf (einmalig vor allen Tests)]")
        plain.info("─" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

        plain.info(f"[SETUP-LOGGING: Referenzlauf abgeschlossen – Referenzzeit: {cls.REFERENCE_TIME:.4f} sec]\n")

    # ------------------------------------------------
    # TESTFALL 1 – Vorhersageprüfung
    # ------------------------------------------------
    def test_1_predict_function(self):
        plain.info("=" * 70)
        plain.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)

        acc, metrics_text = evaluate_model(model, X_test, y_test)

        # Technische Logs und Metriken gemeinsam
        plain.info(metrics_text)

        self.assertGreaterEqual(acc, 0.9)
        plain.info("Ergebnis: TESTFALL 1 PASSED ✅\n")

    # ------------------------------------------------
    # TESTFALL 2 – Laufzeitprüfung
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
            plain.info("Ergebnis: TESTFALL 2 FAILED ❌\n")
            raise


if __name__ == "__main__":
    plain.info("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
