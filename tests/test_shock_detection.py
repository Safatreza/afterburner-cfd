import unittest
from src.rayleigh_solver import detect_shock

class TestShockDetection(unittest.TestCase):
    def test_detect_shock(self):
        # Mach profile with a shock at index 5
        mach_profile = [1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3]
        shock_index = detect_shock(mach_profile)
        self.assertEqual(shock_index, 5)

if __name__ == '__main__':
    unittest.main() 