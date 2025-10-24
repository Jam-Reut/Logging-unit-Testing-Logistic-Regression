# test_logistic_model.py

import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# --- technischer Logger (mit Zeitstempel) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# --- Plain-Logger (ohne Zeitstempel) für Headline/Metriken ---
plain = logging.getLogger("plain")
_plain_handler = logging.StreamHandler()
_plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain.addHandler(_plain_handler)
plain.setLevel(logging.INFO)
plain.propagate = False


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        # EINZIGER Start-Banner
        plain.info("\n=== Starte Unit-Tests ===\n")

        # Header direkt vor dem gesamten Block
        plain.info("=" * 70)
        plain.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain.info("=" * 70 + "\n")

        # Keine Marker dazwischen -> zusammenhängender evaluate-Block
        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9)
        plain.info("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        plain.info("=" * 70)
        plain.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        plain.info("=" * 70 + "\n")

        plain.info("[TEST 2 LOGGING: Referenzlauf (einmalig vor TESTFALL 2)]\n")
        plain.info("[TEST 2 LOGGING: Referenzlauf beginnt]\n")

        df = load_data("advertising.csv")
        train_model(df)
        ref_time = get_last_timing("train_model")
        plain.info(f"[TEST 2 LOGGING: Referenzlauf abgeschlossen – {ref_time:.4f} sec]\n")

        plain.info("[TEST 2 LOGGING: aktueller Lauf (im Unittest) – beginnt]\n")
        train_model(df)
        runtime = get_last_timing("train_model")
        plain.info(f"[TEST 2 LOGGING: aktueller Lauf abgeschlossen – {runtime:.4f} sec]\n")

        limit = ref_time * 1.2
        plain.info("Laufzeitanalyse:")
        plain.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        self.assertLessEqual(runtime, limit)
        plain.info("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
