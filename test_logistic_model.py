# ======================================================
# Unit Tests für Logistic Regression Modell
# Testfall 1: predict() -> Accuracy & Confusion Matrix
# Testfall 2: fit() -> Laufzeit ≤ 120 % der Referenz
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# ------------------------------------------------------
# Logging Setup (Anhängen an das zentrale Logfile)
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
    # SETUP
    # --------------------------------------------------
    @classmethod
    def setUpClass(cls):
        print("\n===================================================")
        print("   Starte Test-Suite für Logistic Regression Modell")
        print("===================================================\n")

        logging.info("=" * 70)
        logging.info(">>> TESTSESSION START")
        logging.info("=" * 70)

        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

        print("Setup abgeschlossen – Modell initial trainiert.\n")
        logging.info("SETUP abgeschlossen\n")

    # --------------------------------------------------
    # TESTFALL 1: Prüfung der Vorhersagefunktion predict()
    # --------------------------------------------------
    def test_predict_function(self):
        """
        Testfall 1:
        Prüft, dass die Vorhersagefunktion predict() korrekt arbeitet.
        Indikatoren: Accuracy >= 0.9 und Confusion Matrix (siehe Log).
        """
        print("===================================================")
        print(" TESTFALL 1: Vorhersagefunktion (predict)")
        print("===================================================\n")

        logging.info("=" * 70)
        logging.info(">>> START TESTFALL 1: Vorhersagefunktion (predict)")
        logging.info("=" * 70)

        # Evaluierung starten
        start = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        runtime = time.perf_counter() - start

        print(f"evaluate_model ran in: {runtime:.4f} sec")
        print(f"Erzielte Accuracy: {accuracy:.3f}\n")

        logging.info(f"TESTFALL 1: Laufzeit evaluate_model = {runtime:.4f} sec")
        logging.info(f"TESTFALL 1: Accuracy = {accuracy:.3f}")

        self.assertGreaterEqual(
            accuracy, 0.9, "Accuracy unter 0.9 – Modellvorhersage nicht ausreichend."
        )

        print("Ergebnis: Testfall 1 PASSED\n")
        logging.info("TESTFALL 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Überprüfung der Laufzeit der Trainingsfunktion fit()
    # --------------------------------------------------
    def test_fit_runtime(self):
        """
        Testfall 2:
        Prüft, dass die Laufzeit der Trainingsfunktion fit()
        ≤ 120 % der Referenzlaufzeit bleibt.
        """
        print("===================================================")
        print(" TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        print("===================================================\n")

        logging.info("=" * 70)
        logging.info(">>> START TESTFALL 2: Laufzeit der Trainingsfunktion (fit)")
        logging.info("=" * 70)

        # Referenzlaufzeit
        t0 = time.perf_counter()
        _ = train_model(self.df)
        ref_time = time.perf_counter() - t0
        print(f"Referenzlaufzeit train_model: {ref_time:.4f} sec")

        # Testlaufzeit
        t1 = time.perf_counter()
        _ = train_model(self.df)
        test_time = time.perf_counter() - t1
        print(f"Aktuelle Laufzeit train_model: {test_time:.4f} sec")

        # Vergleich
        limit = ref_time * 1.2
        print(f"Zulässiges Limit (120 %): {limit:.4f} sec\n")

        print("Laufzeitanalyse:")
        print(f" - Referenzlaufzeit : {ref_time:.4f} sec")
        print(f" - Aktuelle Laufzeit: {test_time:.4f} sec")
        print(f" - Erlaubtes Limit  : {limit:.4f} sec\n")

        logging.info(f"TESTFALL 2: Referenzlaufzeit = {ref_time:.4f} sec")
        logging.info(f"TESTFALL 2: Aktuelle Laufzeit = {test_time:.4f} sec")
        logging.info(f"TESTFALL 2: Zulässiges Limit (120 %) = {limit:.4f} sec")

        self.assertLessEqual(
            test_time,
            limit,
            f"Laufzeit {test_time:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )

        print("Ergebnis: Testfall 2 PASSED\n")
        logging.info("TESTFALL 2 PASSED\n")

    # --------------------------------------------------
    # TEARDOWN
    # --------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        print("===================================================")
        print("   Alle Testfälle erfolgreich abgeschlossen.")
        print("===================================================\n")

        logging.info("=" * 70)
        logging.info(">>> TESTSESSION ENDE")
        logging.info("=" * 70)


# ------------------------------------------------------
# Testausführung
# ------------------------------------------------------
if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
