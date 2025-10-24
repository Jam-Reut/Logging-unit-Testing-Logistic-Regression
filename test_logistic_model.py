import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


# Referenzlaufzeit wird beim ersten Import dynamisch ermittelt
_df_ref = load_data("advertising.csv")
train_model(_df_ref)
REFERENCE_TIME = get_last_timing("train_model")

# Falls du manuell testen willst, hier aktivieren:
# REFERENCE_TIME = 0.3971


class TestLogisticRegressionModel(unittest.TestCase):

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        print()
        print("=" * 54)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 54)

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("=" * 54)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 54)
        print()

        df = load_data("advertising.csv")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")

        # Vergleich mit dynamisch bestimmter (oder fester) Referenzzeit
        ref_time = REFERENCE_TIME
        limit = ref_time * 1.2

        print("Laufzeitanalyse, um die gemessenen Zeiten nachvollziehen zu können:")
        print(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime <= limit:
            print("Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            print("❌ Laufzeit überschreitet das Limit!\n")

        self.assertLessEqual(runtime, limit)
        print("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
