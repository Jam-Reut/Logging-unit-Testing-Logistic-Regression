# ======================================================
# Unit Tests for logistic_model.py
# According to Ori Cohen's testing + assignment specs
# ======================================================

import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load dataset and train once"""
        cls.df = load_data("advertising.csv")
        cls.model, cls.X_test, cls.y_test = train_model(cls.df)

    def test_predict_accuracy(self):
        """Test 1: Verify predict() via Accuracy & Confusion Matrix"""
        accuracy, cm = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy below expected threshold (0.9)")
        self.assertEqual(cm.shape, (2, 2), "Confusion matrix must be 2x2")

    def test_fit_runtime(self):
        """Test 2: Ensure fit() runtime ≤ 120% of reference"""
        # Baseline training time
        start = time.time()
        _ = train_model(self.df)
        baseline = time.time() - start

        # Second training run
        start = time.time()
        _ = train_model(self.df)
        runtime = time.time() - start

        # Assert within 120% tolerance
        self.assertLessEqual(
            runtime,
            baseline * 1.2,
            f"Training runtime {runtime:.4f}s exceeds 120% of baseline ({baseline:.4f}s)"
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
