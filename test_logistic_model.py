# test_logistic_model.py

import unittest
import logging
from logistic_model import (
    load_data, train_model, evaluate_model,
    get_last_timing, get_last_metrics_text
)

# --- technischer Logger (mit Zeitstempel) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# --- Plain-Logger (ohne Zeitstempel) für Überschriften/Metriken ---
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
        plain.info("\n=== Starte Unit-Tests ===\n")

        plain.info("=" * 70)
        plain.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain.info("=" * 70 + "\n")

        # Erst der komplette technische Log-Block (load_data -> train_model -> evaluate_model)
        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        # Dann gesammelt und in einem Stück die Metriken ausgeben
        #plain.info("\n[TEST 1 LOGGING: Vorhersageprüfung]\n")
        plain.info(" \n")
        plain.info(get_last_metrics_text())

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
        #plain.info("[TEST 2 LOGGING: Referenzlauf beginnt]\n")

        df = load_data("advertising.csv")
        train_model(df)
        ref_time = get_last_timing("train_model")
        #plain.info(f"[TEST 2 LOGGING: Referenzlauf abgeschlossen – {ref_time:.4f} sec]\n")

        #plain.info("[TEST 2 LOGGING: aktueller Lauf beginnt]\n")
        plain.info("[TEST 2 LOGGING: aktuelle Laufzeit]\n")
        train_model(df)
        runtime = get_last_timing("train_model")
        #plain.info(f"[TEST 2 LOGGING: aktueller Lauf abgeschlossen – {runtime:.4f} sec]\n")

        limit = ref_time * 1.2
        plain.info("Laufzeitanalyse:")
        plain.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        self.assertLessEqual(runtime, limit)
        plain.info("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
