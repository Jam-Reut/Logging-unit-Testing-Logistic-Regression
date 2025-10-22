import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("=" * 54)
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("=" * 54)
        print("Starte Testfall 1 – Validierung der Modellvorhersage...\n")

        df = load_data("advertising.csv")
        print(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")

        model, X_test, y_test = train_model(df)
        print(f"→ train_model ran in: {get_last_timing('train_model'):.4f} sec")

        accuracy = evaluate_model(model, X_test, y_test)
        print(f"→ evaluate_model ran in: {get_last_timing('evaluate_model'):.4f} sec")

        print(f"\nAccuracy: {accuracy:.3f}")
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("=" * 54)
        print("TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("=" * 54)
        print("Starte Testfall 2 – A
