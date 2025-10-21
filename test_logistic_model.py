# ======================================================
# Unit Tests (extended logging)
# ======================================================

import unittest
import time
import logging
from logistic_model import load_data, train_model, evaluate_model


# Configure a separate test log
logging.basicConfig(
    filename="test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class TestLogisticModel(unittest.TestCase):
    """Automated ML tests with extensive logging."""

    @classmethod
    def setUpClass(cls):
        logging.info("=== 🧪 Starting ML Unit Tests ===")
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)
        logging.info("Setup completed: Model trained successfully.")

    def test_predict_accuracy(self):
        """Test 1: Validate predict() using Accuracy and Confusion Matrix"""
        logging.info("▶️ Running test_predict_accuracy()")
        accuracy, cm = evaluate_model(self.model, self.X_test, self.y_test)
        logging.info(f"✅ Accuracy achieved: {accuracy:.3f}")
        logging.info(f"✅ Confusion Matrix:\n{cm}")
        self.assertGreaterEqual(accuracy, 0.9, "❌ Accuracy is below 0.9 threshold.")
        self.assertEqual(cm.shape, (2, 2), "❌ Confusion Matrix is not 2x2.")
        logging.info("✅ test_predict_accuracy PASSED\n")

    def test_fit_runtime(self):
        """Test 2: Check training runtime stability (≤130% of baseline)"""
        logging.info("▶️ Running test_fit_runtime()")

        # 1️⃣ Baseline runtime
        start = time.time()
        _ = train_model(self.df)
        baseline = time.time() - start
        logging.info(f"Baseline runtime: {baseline:.4f} sec")

        # 2️⃣ Second run
        start = time.time()
        _ = train_model(self.df)
        runtime = time.time() - start
        logging.info(f"Second runtime: {runtime:.4f} sec")

        # 3️⃣ Compare with 130% tolerance
        limit = baseline * 1.3
        logging.info(f"Allowed limit (130% of baseline): {limit:.4f} sec")

        if runtime <= limit:
            logging.info("✅ test_fit_runtime PASSED")
        else:
            logging.warning(
                f"⚠️ Training runtime {runtime:.4f}s exceeded limit {limit:.4f}s"
            )

        self.assertLessEqual(
            runtime,
            limit,
            f"Laufzeit {runtime:.4f}s überschreitet 130 % der Referenzzeit ({baseline:.4f}s)"
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
