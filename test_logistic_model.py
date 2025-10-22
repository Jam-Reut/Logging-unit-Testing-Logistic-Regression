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

        # Datensatz laden und Modell trainieren
        df = load_data("advertising.csv")
        model, X_test, y_test = train_model(df)

        # Modell evaluieren
        accuracy = evaluate_model(model, X_test, y_test)

        # Prüfung
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
        print("Starte Testfall 2 – Analyse der Trainingslaufzeit...\n")

        df = load_data("advertising.csv")

        # 1️⃣ Referenzlauf (Baseline)
        print("=== Modelltraining (Referenzlauf) ===")
        train_model(df)
        ref_time = get_last_timing("train_model")  # Nur Laufzeit holen, keine doppelte Ausgabe!
        print(f"→ train_model ran in: {ref_time:.4f} sec\n")

        # 2️⃣ Testlauf
        print("=== Modelltraining (Testlauf) ===")
        train_model(df)
        runtime = get_last_timing("train_model")
        print(f"→ train_model ran in: {runtime:.4f} sec\n")

        limit = ref_time * 1.2

        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        try:
            self.assertLessEqual(runtime, limit)
            print("Ergebnis: TESTFALL 2 PASSED\n")
        except AssertionError:
            print("Ergebnis: TESTFALL 2 FAILED ❌\n")
            raise


if __name__ == "__main__":
    # print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
