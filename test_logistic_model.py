import unittest
import time
from logistic_model import load_data, train_model, evaluate_model


class TestLogisticModel(unittest.TestCase):
    def setUp(self):
        self.df = load_data("advertising.csv")
        self.model, self.X_test, self.y_test = train_model(self.df)

    def test_predict_function(self):
        accuracy = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy ist zu niedrig (< 0.9)")

    def test_fit_runtime(self):
        start = time.time()
        _ = train_model(self.df)
        ref_time = time.time() - start

        start = time.time()
        _ = train_model(self.df)
        runtime = time.time() - start

        self.assertLessEqual(
            runtime,
            ref_time * 1.2,
            f"Laufzeit {runtime:.4f}s überschreitet 120 % der Referenzzeit ({ref_time:.4f}s)"
        )


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
