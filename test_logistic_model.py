import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# ------------------------------------------------------------
# Technischer Logger (mit Zeitstempel)
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# ------------------------------------------------------------
# Zweiter Logger (ohne Zeitstempel für Marker)
# ------------------------------------------------------------
plain_logger = logging.getLogger("plain")
plain_handler = logging.StreamHandler()
plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(plain_handler)
plain_logger.propagate = False


class TestLogisticRegressionModel(unittest.TestCase):

    def test_1_predict_function(self):
        # Header ganz oben
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain_logger.info("=" * 70 + "\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc, metrics_output = evaluate_model(model, X_test, y_test)

        # Ergebnisse (Accuracy usw.)
        plain_logger.info(metrics_output)

        # Marker vor Logs
        plain_logger.info("[TEST 1 LOGGING: Vorhersageprüfung]\n")

        self.assertGreaterEqual(acc, 0.9)
        plain_logger.info("Ergebnis: TESTFALL 1 PASSED\n")

        # Direkt Start von Testfall 2
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        plain_logger.info("=" * 70 + "\n")

    def test_2_train_runtime(self):
        plain_logger.info("─" * 70)
        plain_logger.info("[TEST 2 LOGGING: Referenzlauf (einmalig vor TESTFALL 2)]")
        plain_logger.info("─" * 70 + "\n")

        plain_logger.info("[TEST 2 LOGGING: Referenzlauf beginnt]\n")
        df = load_data("advertising.csv")
        train_model(df)
        ref_time = get_last_timing("train_model")
        plain_logger.info(f"[TEST 2 LOGGING: Referenzlauf abgeschlossen – {ref_time:.4f} sec]\n")

        plain_logger.info("[TEST 2 LOGGING: aktueller Lauf (im Unittest) – beginnt]\n")
        train_model(df)
        runtime = get_last_timing("train_model")
        plain_logger.info(f"[TEST 2 LOGGING: aktueller Lauf abgeschlossen – {runtime:.4f} sec]\n")

        limit = ref_time * 1.2
        plain_logger.info("Laufzeitanalyse:")
        plain_logger.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain_logger.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain_logger.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime <= limit:
            plain_logger.info("Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            plain_logger.info("❌ Laufzeit überschreitet das Limit!\n")

        self.assertLessEqual(runtime, limit)
        plain_logger.info("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    plain_logger.info("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
