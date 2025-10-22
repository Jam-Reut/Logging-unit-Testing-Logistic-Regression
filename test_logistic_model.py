import unittest
from logistic_model import load_data, train_model, evaluate_model, get_last_timing


class TestLogisticRegressionModel(unittest.TestCase):
    """
    Diese Testklasse überprüft ein ML-Modell (Logistic Regression)
    gemäß den Prinzipien aus O. Cohens Artikel:
    „Unit Testing and Logging for Data Science“.
    
    Ziel:
      - Testfall 1: Validierung der Vorhersagefunktion (predict)
                    über Accuracy & Confusion Matrix
      - Testfall 2: Prüfung der Laufzeitstabilität der Trainingsfunktion (fit)
                    mit einer Toleranz von 120 % der Referenzzeit
    """

    @classmethod
    def setUpClass(cls):
        """Setup wird nur einmal zu Beginn ausgeführt, nicht vor jedem Test."""
        #print("\n=== Starte Unit-Tests ===")
        print("Setup: Initiales Training wird ausgeführt...\n")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print("Initiales Setup abgeschlossen.\n")

    # ------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # ------------------------------------------------
    def test_1_predict_function(self):
        print("=" * 54)
        print("TESTFALL 1: Vorhersagefunktion (predict)")
        print("=" * 54)
        print("Starte Testfall 1 – Validierung der Modellvorhersage...\n")

        accuracy = evaluate_model(self.model, self.X_test, self.y_test)

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

        # Referenzlauf (Baseline)
        _ = train_model(self.df)
        ref_time = get_last_timing("train_model")

        # Zweiter Lauf (Testlauf)
        _ = train_model(self.df)
        runtime = get_last_timing("train_model")

        # 120 % Toleranzgrenze
        limit = ref_time * 1.2

        # Analyse soll IMMER ausgegeben werden (auch bei Fehler)
        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        # Bedingung prüfen
        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("Ergebnis: TESTFALL 2 PASSED\n")


        print("=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit: {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit: {runtime:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
