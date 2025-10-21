# ======================================================
# Unit Tests für Logistic Regression Modell
# Testfall 1: predict() -> Accuracy & Confusion Matrix
# Testfall 2: fit() -> Laufzeit ≤ 120 % der Referenz
# Ausgabe kompakt (Ori Cohen-Stil) + klare Testfall-Markierungen
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# Zentrales Logfile (gleich wie in logistic_model.py)
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    filemode="a"  # an bestehende Logs anhängen
)


class TestLogisticRegressionModel(unittest.TestCase):
    """Automatisierte Tests gemäß Aufgabenstellung."""

    # --------------------------------------------------
    # SETUP: Daten laden & Modell initial trainieren
    # --------------------------------------------------
    @classmethod
    def setUpClass(cls):
        print("\n.setUpClass  Initialisiere Testumgebung und trainiere initiales Modell...")
        logging.info("=" * 70)
        logging.info("TESTSESSION START")
        logging.info("=" * 70)
        logging.info("SETUP: Lade Daten und trainiere initiales Modell (für Testfall 1)")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        print(".setUpClass completed.\n")
        logging.info("SETUP: abgeschlossen\n")

    # --------------------------------------------------
    # TESTFALL 1: Prüfung der Vorhersagefunktion predict()
    # --------------------------------------------------
    def test_predict_function(self):
        """
        Testfall 1:
        Prüft, dass die Vorhersagefunktion predict() korrekt arbeitet.
        Indikatoren: Accuracy >= 0.9 und Confusion Matrix (wird im Logging ausgegeben).
        """
        logging.info("-" * 70)
        logging.info("TESTFALL 1 START: Prüfung der Vorhersagefunktion predict()")
        logging.info("-" * 70)

        print("Running Testfall 1: Prüfung der Vorhersagefunktion predict()")
        t0 = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        dt = time.perf_counter() - t0
        print(f".evaluate_model ran in: {dt:.4f} sec")

        logging.info(f"TESTFALL 1: Accuracy = {accuracy:.3f}")
        logging.info(f"TESTFALL 1: Dauer evaluate_model = {dt:.4f} sec")

        self.assertGreaterEqual(
            accuracy, 0.9, "Accuracy unter 0.9 – Modellvorhersage nicht ausreichend."
        )

        logging.info("TESTFALL 1 PASSED\n")
        print("Testfall 1 PASSED\n")

    # --------------------------------------------------
    # TESTFALL 2: Überprüfung der Laufzeit der Trainingsfunktion fit()
    # --------------------------------------------------
    def test_fit_runtime(self):
        """
        Testfall 2:
        Prüft, dass die Laufzeit der Trainingsfunktion fit()
        ≤ 120 % der Referenzlaufzeit bleibt.
        """
        logging.info("-" * 70)
        logging.info("TESTFALL 2 START: Überprüfung der Laufzeit der Trainingsfunktion fit()")
        logging.info("-" * 70)

        print("Running Testfall 2: Überprüfung der Laufzeit der Trainingsfunktion fit()")

        # Referenzlauf
        t0 = time.perf_counter()
        _ = train_model(self.df)
        baseline = time.perf_counter() - t0
        print(f".train_model (Referenzlauf) ran in: {baseline:.4f} sec")
        logging.info(f"TESTFALL 2: Referenzlaufzeit = {baseline:.4f} sec")

        # Testlauf
        t0 = time.perf_counter()
        _ = train_model(self.df)
        runtime = time.perf_counter() - t0
        print(f".train_model (Testlauf) ran in: {runtime:.4f} sec")
        logging.info(f"TESTFALL 2: Aktuelle Laufzeit = {runtime:.4f} sec")

        # Analyse
        limit = baseline * 1.2
        print(f".Laufzeitanalyse → Limit (120 %): {limit:.4f} sec")
        print(f".Vergleich: aktuelle Laufzeit = {runtime:.4f} sec | Referenz = {baseline:.4f} sec\n")
        logging.info(f"TESTFALL 2: Erlaubtes Limit (120 %) = {limit:.4f} sec")

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({baseline:.4f}s)"
        )

        logging.info("TESTFALL 2 PASSED\n")
        print("Testfall 2 PASSED\n")

    # --------------------------------------------------
    # TEARDOWN
    # --------------------------------------------------
    @classmethod
    def tearDownClass(cls):
        logging.info("=" * 70)
        logging.info("TESTSESSION ENDE")
        logging.info("=" * 70)
        print(".tearDownClass  Alle Testfälle abgeschlossen.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
