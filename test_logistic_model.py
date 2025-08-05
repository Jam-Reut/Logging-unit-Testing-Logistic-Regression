import unittest
import logistic_model as lm
import pandas as pd

class TestLogisticModel(unittest.TestCase):

    def setUp(self):
        self.df = lm.load_data("advertising.csv")

    def test_data_loaded(self):
        """Test ob Daten geladen werden"""
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertFalse(self.df.empty)

    def test_model_training(self):
        """Test ob Modell trainiert und Vorhersagen erzeugt"""
        model, X_test, y_test = lm.train_model(self.df)
        preds = model.predict(X_test)
        self.assertEqual(len(preds), len(y_test))

if __name__ == "__main__":
    unittest.main()
