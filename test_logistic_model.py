import unittest
import logging
from logistic_model import load_data, train_model, evaluate_model, get_last_timing

# Logging so konfigurieren, dass alles in denselben Stream läuft
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True
)

class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        logging.info("──────────────────────────────────────────────────────────────")
        logging.info("[TEST 2 LOGGING: Referenzlauf (einmalig vor allen Tests)]")
        logging.info("──────────────────────────────────────────────────────────────")

        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

        logging.info(f"Referenzlauf abgeschlossen – Referenzzeit: {cls.REFERENCE_TIME:.4f} sec")

    def test_1_predict_function(self):
        logging.info("======================================================================")
        logging.info("TESTFALL 1: predict(): Vorhersagefunktion")
        logging.info("======================================================================")
        logging.info("[TEST 1 LOGGING: Vorhersageprüfung]")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)
        self.assertGreaterEqual(acc, 0.9)
        logging.info("Ergebnis: TESTFALL 1 PASSED")

    def test_2_train_runtime(self):
        logging.info("======================================================================")
        logging.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        logging.info("======================================================================")
        logging.info("[TEST 2 LOGGING: aktueller Lauf (im Unittest)]")

        df = load_data("advertising.csv")
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
        limit = ref_time * 1.2

        logging.info("Laufzeitanalyse, um die gemessenen Zeiten nachvollziehen zu können:")
        logging.info(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        logging.info(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        logging.info(f" - Erlaubtes Limit (120%): {limit:.4f} sec")

        if runtime <= limit:
            logging.info("Laufzeit liegt innerhalb der Toleranz.")
        else:
            logging.info("❌ Laufzeit überschreitet das Limit!")

        self.assertLessEqual(runtime, limit)
        logging.info("Ergebnis: TESTFALL 2 PASSED")


if __name__ == "__main__":
    logging.info("=== Starte Unit-Tests ===")
    unittest.main(argv=[""], exit=False)
