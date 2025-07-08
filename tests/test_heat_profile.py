import unittest
from src.heat_profile_plugin import heat_profile

class TestHeatProfile(unittest.TestCase):
    def test_heat_profile_peak(self):
        x = [i * 0.1 for i in range(11)]
        peak = 1e6
        width = 0.5
        profile = heat_profile(x, peak, width)
        max_value = max(profile)
        self.assertAlmostEqual(max_value, peak, delta=peak*0.01)
        self.assertEqual(profile[len(x)//2], max_value)

if __name__ == '__main__':
    unittest.main() 