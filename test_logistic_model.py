import unittest
import contextlib
import io
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

        # Datensatz laden
        df = load_data("advertising.csv")
        print(f"→ load_data ran in: {get_last_timing('load_data'):.4f} sec")

        # Modell trainieren
        model, X_test, y_test = train_model(df)
        print(f"→ train_model ran in: {get_last_timing('train_model'):.4f} sec")

        # Modell evaluieren
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
        print("Starte Testfall 2 – Analyse der Trainingslaufzeit...\n")

        # stdout umleiten (wir wollen keine Ausgaben aus logistic_model)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = load_data("advertising.csv")

            # Referenzlauf
            train_model(df)
            ref_time = get_last_timing("train_model")

            # Testlauf
            train_model(df)
            runtime = get_last_timing("train_model")

        # Jetzt geben wir die interne Laufzeit-Analyse sichtbar aus
        limit = ref_time * 1.2
        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            runtime, limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )
        print("Ergebnis: TESTFALL 2 PASSED\n")

        # Optional: wenn du willst, kannst du die internen Ausgaben aus logistic_model hier anzeigen:
        # print("=== Interner Output (logistic_model) ===")
        # print(buf.getvalue())


if __name__ == "__main__":
    print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
