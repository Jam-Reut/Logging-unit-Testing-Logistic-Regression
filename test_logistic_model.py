import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ----------------------------------------------------------
        # Referenzlauf wird EINMAL vor allen Tests durchgeführt
        # ----------------------------------------------------------
        print("\n" + "=" * 70)
        print("[TEST 2 LOGGING: Referenzlauf (einmalig vor allen Tests)]")
        print("=" * 70 + "\n")

        df = load_data("advertising.csv")
        train_model(df)
        cls.REFERENCE_TIME = get_last_timing("train_model")

        print(f"Referenzlauf abgeschlossen – Referenzzeit: {cls.REFERENCE_TIME:.4f} sec\n")

        # Falls du mit fester Referenzzeit testen willst:
        # cls.REFERENCE_TIME = 0.3971

    # ------------------------------------------------
    # TESTFALL 1: predict(): Vorhersagefunktion
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("\n" + "=" * 70)
        print("TESTFALL 1: predict(): Vorhersagefunktion")
        print("=" * 70)
        print("\n[TEST 1 LOGGING: Vorhersageprüfung]\n")

        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)
        acc = evaluate_model(model, X_test, y_test)

        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("\nErgebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("\n" + "=" * 70)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 70)
        print("\n[TEST 2 LOGGING: aktueller Lauf (im Unittest)]\n")

        df = load_data("advertising.csv")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")

        ref_time = self.REFERENCE_TIME
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
    #print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
