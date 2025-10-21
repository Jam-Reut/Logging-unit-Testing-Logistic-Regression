# ======================================================
# Unit Tests for Logistic Regression Pipeline
# Ori Cohen–style: Accuracy & 120%-Runtime tests
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model

# Central test log
logging.basicConfig(
    filename="test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestLogisticModel(unittest.TestCase):
    """Automated ML Unit Tests with detailed logging."""

    @classmethod
    def setUpClass(cls):
        logging.info("🚀 Setting up: loading data and training model once.")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

    def test_predict_accuracy(self):
        """Test 1: Validate predict() using Accuracy and Confusion Matrix."""
        logging.info("▶️ Running test_predict_accuracy()")
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        logging.info(f"✅ Accuracy achieved: {accuracy:.3f}")
        self.assertGreaterEqual(accuracy, 0.9, "❌ Accuracy below 0.9 threshold.")
        logging.info("✅ test_predict_accuracy PASSED\n")

    def test_fit_runtime(self):
        """Test 2: Verify train_model runtime ≤ 120% of baseline."""
        logging.info("▶️ Running test_fit_runtime()")

        # Baseline training time
        start = time.perf_counter()
        _ = train_model(self.df)
        baseline = time.perf_counter() - start
        logging.info(f"Baseline runtime: {baseline:.4f} sec")

        # Second training
        start = time.perf_counter()
        _ = train_model(self.df)
        runtime = time.perf_counter() - start
        logging.info(f"Current runtime: {runtime:.4f} sec")

        # Compare with 120% tolerance (strict requirement)
        limit = baseline * 1.2
        logging.info(f"Allowed limit (120%): {limit:.4f} sec")

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({baseline:.4f}s)"
        )
        logging.info("✅ test_fit_runtime PASSED\n")


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
