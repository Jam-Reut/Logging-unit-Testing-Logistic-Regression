# ======================================================
# UNIT TESTS für Logistic Regression Modell
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model


# ------------------------------------------------------
# LOGGING (Anhängen an zentrales Logfile)
# ------------------------------------------------------
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(message)s",
    filemode="a"
)


class TestLogisticRegressionModel(unittest.TestCase):
    """Automatisierte Tests für das ML-System."""

    # --------------------------------------------------
    # SETUP (einmalig für alle Tests)
    # --------------------------------------------------
    @classmethod
    def setUpClass(cls):
        print("# ======================================================")
        print("# UNIT TEST AUSFÜHRUNG (test_logistic_model.py)")
        print("# ======================================================\n")

        print("Starte Test-Suite für Logistic Regression Modell\n")

        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

        print("Setup abgeschlossen – Modell initial trainiert.\n")

    # --------------------------------------------------
    # TESTFALL 1: Vorhersagefunktion (predict)
    # --------------------------------------------------
    def test_1_predict_function(self):
        """Testfall 1: Accuracy ≥ 0.9 und Confusion Matrix vorhanden."""
        print("=== Testfall 1: Vorhersagefunktion (predict) ===")
        start = time.perf_counter()

        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start

        print(f"  → evaluate_model ran in: {runtime:.4f} sec")
        print(f"  Accuracy: {accuracy:.3f}\n")

        self.assertGreaterEqual(
            accuracy, 0.9, "Accuracy unter 0.9 – Modellvorhersage nicht ausreichend."
        )

        print("  ✅ Ergebnis: Testfall 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Laufzeit der Trainingsfunktion (fit)
    # --------------------------------------------------
    def test_2_fit_runtime(self):
        """Testfall 2: Laufzeit ≤ 120 % der Referenz."""
        print("=== Testfall 2: Laufzeit der Trainingsfunktion (fit) ===")

        # 1️⃣ Referenzlaufzeit (erster Trainingslauf)
        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0

        # 2️⃣ Testlaufzeit (zweiter Trainingslauf)
        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1

        # 3️⃣ Vergleich mit 120%-Grenze
        limit = ref_time * 1.2

        print(f"\n=== Laufzeit-Analyse ===")
        print(f"  Referenzlaufzeit (erster Trainingslauf): {ref_time:.4f} sec")
        print(f"  Aktuelle Laufzeit (zweiter Trainingslauf): {test_time:.4f} sec")
        print(f"  Erlaubtes Limit (120 %): {limit:.4f} sec\n")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("  ✅ Ergebnis: Testfall 2 PASSED\n")

    # --------------------------------------------------
    # TEARDOWN (am Ende der Tests)
    # --------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        print("# ======================================================")
        print("# TESTERGEBNISSE")
        print("# ======================================================\n")
        print("Alle Testfälle abgeschlossen.\n")


# ------------------------------------------------------
# TESTAUSFÜHRUNG
# ------------------------------------------------------
if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
