import unittest
from src.rayleigh_solver import RayleighFlowSolver

class TestRayleighFlowSolver(unittest.TestCase):
    def test_solve(self):
        config = {
            'mach': 1.5,
            'pressure': 101325,
            'temperature': 300,
            'heat_peak': 1e6,
            'heat_width': 0.1,
            'length': 1.0,
            'points': 10
        }
        solver = RayleighFlowSolver(config)
        results = solver.solve()
        self.assertIsNotNone(results)
        self.assertIn('mach', results)
        self.assertEqual(len(results['mach']), config['points'])

if __name__ == '__main__':
    unittest.main() 