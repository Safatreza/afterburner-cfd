import csv
from pathlib import Path
from typing import Dict, Any
import numpy as np
import multiprocessing as mp
from compressible_ns_solver import CompressibleNSSolver

class SimulationConfig:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the simulation configuration with validation."""
        self.mach = self._validate_float(config.get('mach', 1.2), 'mach', 1.0, 1.5)
        self.pressure = self._validate_float(config.get('pressure', 101325.0), 'pressure', 0.0)
        self.temperature = self._validate_float(config.get('temperature', 300.0), 'temperature', 0.0)
        self.heat_peak = self._validate_float(config.get('heat_peak', 1e6), 'heat_peak', 0.0)
        self.heat_width = self._validate_float(config.get('heat_width', 0.1), 'heat_width', 0.0)
        self.length = self._validate_float(config.get('length', 1.0), 'length', 0.0)
        self.points = self._validate_int(config.get('points', 1000), 'points', 100)
        self.export_results = config.get('export', False)
        self.output_path = config.get('output', 'results.csv')
    
    def _validate_float(self, value: float, name: str, min_val: float, max_val: float = None) -> float:
        """Validate a float parameter."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if value < min_val:
            raise ValueError(f"{name} must be greater than {min_val}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be less than {max_val}")
        return float(value)
    
    def _validate_int(self, value: int, name: str, min_val: int) -> int:
        """Validate an integer parameter."""
        if not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < min_val:
            raise ValueError(f"{name} must be greater than {min_val}")
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'mach': self.mach,
            'pressure': self.pressure,
            'temperature': self.temperature,
            'heat_peak': self.heat_peak,
            'heat_width': self.heat_width,
            'length': self.length,
            'points': self.points,
            'export': self.export_results,
            'output': self.output_path
        }

class ResultsExporter:
    def __init__(self):
        """Initialize the results exporter."""
        pass
    
    def export_to_csv(self, results: Dict[str, np.ndarray], output_path: str):
        """Export simulation results to a CSV file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Position (m)', 'Mach Number', 'Pressure (Pa)',
                           'Temperature (K)', 'Density (kg/m³)', 'Velocity (m/s)'])
            
            # Write data
            for i in range(len(results['x'])):
                writer.writerow([
                    results['x'][i],
                    results['mach'][i],
                    results['pressure'][i],
                    results['temperature'][i],
                    results['density'][i],
                    results['velocity'][i]
                ])
        
        print(f"Results exported to {output_path}")
    
    def export_shock_locations(self, shock_locations: np.ndarray, output_path: str):
        """Export shock locations to a separate CSV file."""
        output_path = Path(output_path)
        shock_path = output_path.parent / f"shocks_{output_path.name}"
        
        with open(shock_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Shock Location (m)'])
            for x in shock_locations:
                writer.writerow([x])
        
        print(f"Shock locations exported to {shock_path}")

def run_simulation(config):
    solver = CompressibleNSSolver(config)
    solver.run(n_steps=config.get('n_steps', 100))
    return solver.get_results()

def run_parameter_study_parallel(config_list, n_workers=4):
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(run_simulation, config_list)
    return results
