import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing, logger

class TestLogisticRegressionModel(unittest.TestCase):

    def test_1_predict_function(self):
        logger.info("=" * 54)
        logger.info("TESTFALL 1: predict(): Vorhersagefunktion")
        logger.info("=" * 54)

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        logger.info(f"Accuracy: {acc:.2f}")
        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (<0.9)")
        logger.info("Ergebnis: TESTFALL 1 PASSED\n")

    def test_2_train_runtime(self):
        logger.info("=" * 54)
        logger.info("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        logger.info("=" * 54)

        df = load_data("advertising.csv")

        train_model(df)
        ref_time = get_last_timing("train_model")
        train_model(df)
        runtime = get_last_timing("train_model")

        limit = ref_time * 1.2
        logger.info("=== Laufzeit-Analyse ===")
        logger.info(f"Referenzlaufzeit: {ref_time:.4f} sec")
        logger.info(f"Aktuelle Laufzeit: {runtime:.4f} sec")
        logger.info(f"Erlaubtes Limit (120%): {limit:.4f} sec")

        self.assertLessEqual(runtime, limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)")
        logger.info("Ergebnis: TESTFALL 2 PASSED\n")

if __name__ == "__main__":
    logger.info("=== Starte Unit-Tests ===")
    unittest.main(argv=[""], exit=False)
