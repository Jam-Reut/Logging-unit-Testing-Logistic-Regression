
import unittest
import pandas as pd
from logistic_model import load_data, train_model

class TestLogisticModel(unittest.TestCase):
    def test_load_data(self):
        df = load_data('advertising.csv')
        self.assertFalse(df.empty)

    def test_train_model(self):
        df = pd.read_csv('advertising.csv')
        X = df[['Age', 'Area Income', 'Daily Time Spent on Site', 'Daily Internet Usage']]
        y = df['Clicked on Ad']
        model, X_test, y_test = train_model(X, y)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
