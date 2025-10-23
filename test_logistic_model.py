import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing, logger


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("\n======================================================")
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("======================================================\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        print(f"\nFinal Accuracy: {acc:.2f}\n")
        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("\n======================================================")
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("======================================================\n")

        # Daten laden (einmal reicht)
        df = load_data("advertising.csv")

        # Referenzlauf
        train_model(df)
        ref_time = get_last_timing("train_model")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")
        limit = ref_time * 1.2

        # Laufzeitanalyse — nach dem Logging-Block ausgeben
        print("\n=== Laufzeit-Analyse ===")
        print(f"Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"Erlaubtes Limit (120%): {limit:.4f} sec")

        if runtime <= limit:
            print("\n✅ Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            print("\n❌ Laufzeit überschreitet das Limit!\n")

        # Assertion am Ende, damit Analyse davor sichtbar bleibt
        self.assertLessEqual(
            runtime, limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    #print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
