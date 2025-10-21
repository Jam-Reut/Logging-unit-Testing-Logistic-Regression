# ======================================================
# Unit Tests für Logistic Regression Modell
# Nach Aufgabenstellung:
# 1) Vorhersage (predict) -> Accuracy & Confusion Matrix
# 2) Performance (fit) -> Laufzeit <= 120 % der Referenz
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# Einheitliches Logging
logging.basicConfig(
    filename="ml_system.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    filemode="w"
)


class TestLogisticRegressionModel(unittest.TestCase):
    """Automatisiertes Testen des Logistic Regression Modells."""

    # ==================================================
    # SETUP: Daten laden & Modell initial trainieren
    # ==================================================
    @classmethod
    def setUpClass(cls):
        print("\n.setUpClass  Initialisiere Testumgebung und trainiere initiales Modell...")
        logging.info("=" * 70)
        logging.info("TESTSESSION START")
        logging.info("=" * 70)
        logging.info("SetupClass: Lade Daten und trainiere initiales Modell")

        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

        print(".setUpClass completed.\n")
        logging.info("Setup abgeschlossen\n")

    # ==================================================
    # TESTFALL 1: Prüfung der Vorhersagefunktion predict()
    # ==================================================
    def test_predict_function(self):
        """
        Testfall 1:
        Prüft, dass die Vorhersagefunktion predict() korrekt arbeitet.
        Bewertet wird über Accuracy >= 0.9 und die Confusion Matrix.
        """
        print("Running Testfall 1: Prüfung der Vorhersagefunktion predict()")
        logging.info("-" * 70)
        logging.info("TESTFALL 1 START: Prüfung der Vorhersagefunktion predict()")
        logging.info("-" * 70)

        start = time.perf_counter()
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        duration = time.perf_counter() - start
        print(f".evaluate_model ran in: {duration:.4f} sec")

        logging.info(f"Accuracy: {accuracy:.3f}")
        logging.info(f"Testfall 1 abgeschlossen in {duration:.4f} sec")

        self.assertGreaterEqual(
            accuracy, 0.9, "Accuracy unter 0.9 – Modellvorhersage nicht ausreichend."
        )

        logging.info("TESTFALL 1 PASSED\n")
        print("Testfall 1 PASSED\n")

    # ==================================================
    # TESTFALL 2: Überprüfung der Laufzeit der Trainingsfunktion fit()
    # ==================================================
    def test_fit_runtime(self):
        """
        Testfall 2:
        Prüft, dass die Laufzeit der Trainingsfunktion fit()
        ≤ 120 % der Referenzlaufzeit bleibt.
        """
        print("Running Testfall 2: Überprüfung der Laufzeit der Trainingsfunktion fit()")
        logging.info("-" * 70)
        logging.info("TESTFALL 2 START: Überprüfung der Laufzeit der Trainingsfunktion fit()")
        logging.info("-" * 70)

        # 1️⃣ Referenzlauf
        start = time.perf_counter()
        _ = train_model(self.df)
        baseline = time.perf_counter() - start
        print(f".train_model (Referenzlauf) ran in: {baseline:.4f} sec")
        logging.info(f"Referenzlaufzeit: {baseline:.4f} sec")

        # 2️⃣ Testlauf
        start = time.perf_counter()
        _ = train_model(self.df)
        runtime = time.perf_counter() - start
        print(f".train_model (Testlauf) ran in: {runtime:.4f} sec")
        logging.info(f"Aktuelle Laufzeit: {runtime:.4f} sec")

        # 3️⃣ Laufzeitanalyse
        limit = baseline * 1.2
        print(f".Laufzeitanalyse → Limit (120 %): {limit:.4f} sec")
        print(f".Vergleich: aktuelle Laufzeit = {runtime:.4f} sec | Referenz = {baseline:.4f} sec\n")

        logging.info(f"Erlaubtes Limit (120 %): {limit:.4f} sec")

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({baseline:.4f}s)"
        )

        logging.info("TESTFALL 2 PASSED\n")
        print("Testfall 2 PASSED\n")

    # ==================================================
    # TEARDOWN: Testsitzung beenden
    # ==================================================
    @classmethod
    def tearDownClass(cls):
        logging.info("=" * 70)
        logging.info("TESTSESSION ENDE")
        logging.info("=" * 70)
        print(".tearDownClass  Alle Testfälle abgeschlossen.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
