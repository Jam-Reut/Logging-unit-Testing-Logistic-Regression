# test_logistic_model.py

import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# --- technischer Logger (mit Zeitstempel) für @mytimer-Ausgaben ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

# --- Plain-Logger (ohne Zeitstempel) für Überschriften/Marker/Metriken ---
plain_logger = logging.getLogger("plain")
_plain_handler = logging.StreamHandler()
_plain_handler.setFormatter(logging.Formatter("%(message)s"))
plain_logger.addHandler(_plain_handler)
plain_logger.propagate = False


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        # Startbanner EINMAL hier (Design-Vorgabe)
        plain_logger.info("\n=== Starte Unit-Tests ===\n")

        # Header direkt vor den Metriken
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 1: predict(): Vorhersagefunktion")
        plain_logger.info("=" * 70 + "\n")

        # Marker vor sämtlichen Logs
        plain_logger.info("[TEST 1 LOGGING: Vorhersageprüfung]\n")

        # ── WICHTIG: technische INFO-Logs für Test 1 unterdrücken,
        # damit die Metriken oben stehen (Kritik vom Prof)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        root_logger.setLevel(logging.WARNING)

        try:
            df = load_data("advertising.csv")
            model, X_test, y_test = train_model(df)
            acc = evaluate_model(model, X_test, y_test)
        finally:
            # nach Test 1 wieder auf INFO für Test 2 zurücksetzen
            root_logger.setLevel(previous_level)

        # Prüfbedingung + Leerzeile vor dem Ergebnis (Design)
        self.assertGreaterEqual(acc, 0.9)
        plain_logger.info("\nErgebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        # Header für Test 2
        plain_logger.info("=" * 70)
        plain_logger.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        plain_logger.info("=" * 70 + "\n")

        # Referenzlauf (mit Logs, daher Level INFO aktiv)
        plain_logger.info("[TEST 2 LOGGING: Referenzlauf (einmalig vor TESTFALL 2)]\n")
        plain_logger.info("[TEST 2 LOGGING: Referenzlauf beginnt]\n")

        df = load_data("advertising.csv")
        train_model(df)
        ref_time = get_last_timing("train_model")
        plain_logger.info(f"[TEST 2 LOGGING: Referenzlauf abgeschlossen – {ref_time:.4f} sec]\n")

        # Aktueller Lauf
        plain_logger.info("[TEST 2 LOGGING: aktueller Lauf (im Unittest) – beginnt]\n")
        train_model(df)
        runtime = get_last_timing("train_model")
        plain_logger.info(f"[TEST 2 LOGGING: aktueller Lauf abgeschlossen – {runtime:.4f} sec]\n")

        # Analyse
        limit = ref_time * 1.2
        plain_logger.info("Laufzeitanalyse:")
        plain_logger.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        plain_logger.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        plain_logger.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        # Testbedingung + Abschluss
        self.assertLessEqual(runtime, limit)
        plain_logger.info("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
