# ======================================================
# Unit Tests für Logistic Regression Pipeline
# Nach Aufgabenstellung: Accuracy + Laufzeit (120 %)
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# Test-Logging
logging.basicConfig(
    filename="test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestLogisticModel(unittest.TestCase):
    """Automatisierte Tests mit Logging & Laufzeitanalyse."""

    @classmethod
    def setUpClass(cls):
        logging.info("🚀 Test-Setup: Lade Daten und trainiere Modell einmalig.")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

    def test_predict_accuracy(self):
        """Test 1: Prüft Modellgenauigkeit & Confusion Matrix."""
        logging.info("▶️ Starte Test: test_predict_accuracy()")
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        logging.info(f"✅ Accuracy erreicht: {accuracy:.3f}")
        self.assertGreaterEqual(
            accuracy, 0.9, "❌ Accuracy unter 0.9 – Modell unzureichend."
        )
        logging.info("✅ Test test_predict_accuracy bestanden.\n")

    def test_fit_runtime(self):
        """Test 2: Prüft Trainingslaufzeit ≤ 120 % der Referenz."""
        logging.info("▶️ Starte Test: test_fit_runtime()")

        # 1️⃣ Erster Lauf (Referenzzeit)
        start = time.perf_counter()
        _ = train_model(self.df)
        baseline = time.perf_counter() - start
        logging.info(f"Referenzlaufzeit: {baseline:.4f} sec")

        # 2️⃣ Zweiter Lauf (aktueller Test)
        start = time.perf_counter()
        _ = train_model(self.df)
        runtime = time.perf_counter() - start
        logging.info(f"Aktuelle Laufzeit: {runtime:.4f} sec")

        # 3️⃣ Prüfung ≤ 120 %
        limit = baseline * 1.2
        logging.info(f"Erlaubtes Limit (120 %): {limit:.4f} sec")

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({baseline:.4f}s)"
        )
        logging.info("✅ Test test_fit_runtime bestanden.\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
