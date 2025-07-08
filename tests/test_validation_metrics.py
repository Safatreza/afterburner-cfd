import unittest
from src.validation.validation_metrics import mae, rmse, r2

class TestValidationMetrics(unittest.TestCase):
    def test_mae(self):
        y_true = [1, 2, 3]
        y_pred = [2, 2, 4]
        self.assertAlmostEqual(mae(y_true, y_pred), 0.6667, places=3)

    def test_rmse(self):
        y_true = [1, 2, 3]
        y_pred = [2, 2, 4]
        self.assertAlmostEqual(rmse(y_true, y_pred), 0.8165, places=3)

    def test_r2(self):
        y_true = [1, 2, 3]
        y_pred = [2, 2, 4]
        self.assertAlmostEqual(r2(y_true, y_pred), 0.5, places=2)

if __name__ == '__main__':
    unittest.main() 