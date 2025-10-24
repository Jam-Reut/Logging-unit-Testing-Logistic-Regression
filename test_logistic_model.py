import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


# ------------------------------------------------
# REFERENZLAUF (wird beim Modulimport einmalig ausgeführt)
# ------------------------------------------------
print("\n[TEST 2 LOGGING: Referenzlauf (beim Modulimport)]")
_df_ref = load_data("advertising.csv")
train_model(_df_ref)
REFERENCE_TIME = get_last_timing("train_model")

# Falls du später mit festen Werten testen willst, einfach diese Zeile aktivieren:
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
        print("\n[TEST 1 LOGGING: Vorhersageprüfung]\n")

        # Daten laden
        df = load_data("advertising.csv")

        # Modell trainieren
        model, X_test, y_test = train_model(df)

        # Modell evaluieren
        acc = evaluate_model(model, X_test, y_test)

        # Genauigkeit prüfen
        self.assertGreaterEqual(acc, 0.9, "Accuracy ist zu niedrig (< 0.9)")
        print("Ergebnis: TESTFALL 1 PASSED\n")

    # ------------------------------------------------
    # TESTFALL 2: fit(): Laufzeit der Trainingsfunktion
    # ------------------------------------------------
    def test_2_train_runtime(self):
        print("=" * 54)
        print("TESTFALL 2: fit(): Laufzeit der Trainingsfunktion")
        print("=" * 54)
        print("\n[TEST 2 LOGGING: aktueller Lauf (im Unittest)]\n")

        # Daten laden
        df = load_data("advertising.csv")

        # Testlauf
        train_model(df)
        runtime = get_last_timing("train_model")

        # Vergleich mit Referenzlaufzeit
        ref_time = REFERENCE_TIME
        limit = ref_time * 1.2

        # Analyse ausgeben
        print("Laufzeitanalyse, um die gemessenen Zeiten nachvollziehen zu können:")
        print(f" - Referenzlaufzeit: {ref_time:.4f} sec")
        print(f" - Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f" - Erlaubtes Limit (120%): {limit:.4f} sec\n")

        if runtime <= limit:
            print("Laufzeit liegt innerhalb der Toleranz.\n")
        else:
            print("❌ Laufzeit überschreitet das Limit!\n")

        # Testbedingung
        self.assertLessEqual(runtime, limit)
        print("Ergebnis: TESTFALL 2 PASSED\n")


if __name__ == "__main__":
    #print("\n=== Starte Unit-Tests ===\n")
    unittest.main(argv=[""], exit=False)
