import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing, logger


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        logger.info("=== TESTFALL 1: predict(): Vorhersagefunktion ===")

        # Daten laden
        df = load_data("advertising.csv")

        # Modell trainieren
        model, X_test, y_test = train_model(df)

        # Modell evaluieren
        acc = evaluate_model(model, X_test, y_test)
        logger.info(f"Final Accuracy: {acc:.2f}")

        # Genauigkeit prüfen
        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        logger.info("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        logger.info("=== TESTFALL 2: fit(): Laufzeit der Trainingsfunktion ===")

        # Daten laden
        df = load_data("advertising.csv")

        # Referenzlauf
        train_model(df)
        ref_time = get_last_timing("train_model")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")

        # Laufzeitanalyse (nur als Ausgabe für Prüfer, kein Logging)
        limit = ref_time * 1.2
        print("\n=== Laufzeit-Analyse ===")
        print(f"Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"Erlaubtes Limit (120%): {limit:.4f} sec")

        if runtime <= limit:
            print("\n✅ Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            print("\n❌ Laufzeit überschreitet das Limit!\n")

        # Testbedingung
        self.assertLessEqual(
            runtime, limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    logger.info("=== Starte Unit-Tests ===")
    unittest.main(argv=[""], exit=False)
