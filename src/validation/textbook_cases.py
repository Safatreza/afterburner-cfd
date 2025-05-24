import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Callable
from pathlib import Path
import json
from .validation_metrics import ValidationMetrics, ValidationAnalyzer
from .validation_plots import ValidationPlotter

class TextbookCase:
    """Base class for textbook case validation."""
    def __init__(self, name: str):
        self.name = name
        self.analyzer = ValidationAnalyzer()
        self.plotter = ValidationPlotter()
    
    def get_reference_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Get the reference solution for the textbook case."""
        raise NotImplementedError("Subclasses must implement get_reference_solution")
    
    def validate(self,
                cfd_data: Dict[str, np.ndarray],
                x_coords: np.ndarray,
                save_dir: Optional[str] = None) -> Dict[str, ValidationMetrics]:
        """Validate CFD results against the textbook case."""
        # Get reference solution
        ref_data = self.get_reference_solution(x_coords)
        
        # Compute metrics
        metrics = {}
        for var_name in cfd_data.keys():
            if var_name not in ref_data:
                print(f"Warning: No reference data found for {var_name}")
                continue
            
            metrics[var_name] = self.analyzer.compute_metrics(
                cfd_data[var_name],
                ref_data[var_name],
                variable_name=f"{self.name}_{var_name}"
            )
            
            # Create comparison plots if save_dir is provided
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Plot comparison
                self.plotter.plot_comparison(
                    cfd_data[var_name],
                    ref_data[var_name],
                    x_data=x_coords,
                    variable_name=var_name,
                    save_path=str(save_dir_path / f"{self.name}_{var_name}_comparison.png")
                )
                
                # Plot error distribution
                self.plotter.plot_error_distribution(
                    cfd_data[var_name],
                    ref_data[var_name],
                    variable_name=var_name,
                    save_path=str(save_dir_path / f"{self.name}_{var_name}_error_dist.png")
                )
        
        # Create summary plot if save_dir is provided
        if save_dir:
            self.plotter.plot_metrics_summary(
                metrics,
                save_path=str(Path(save_dir) / f"{self.name}_metrics_summary.png")
            )
        
        return metrics

class LaminarPoiseuilleFlow(TextbookCase):
    """Validation against laminar Poiseuille flow solution."""
    def __init__(self, dp_dx: float, mu: float, rho: float, h: float):
        super().__init__("LaminarPoiseuilleFlow")
        self.dp_dx = dp_dx  # Pressure gradient
        self.mu = mu        # Dynamic viscosity
        self.rho = rho      # Density
        self.h = h          # Channel height
    
    def get_reference_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Get the analytical solution for laminar Poiseuille flow."""
        y = np.array(x)  # Ensure y is an array
        # Velocity profile: u(y) = -dp/dx * (h^2 - y^2) / (2*mu)
        u = -self.dp_dx * (self.h**2 - y**2) / (2 * self.mu)
        # Pressure profile (linear): p(x) = dp/dx * x
        p = self.dp_dx * y
        return {
            'u': u,
            'p': p
        }

class BlasiusBoundaryLayer(TextbookCase):
    """Validation against Blasius boundary layer solution."""
    def __init__(self, u_inf: float, nu: float):
        super().__init__("BlasiusBoundaryLayer")
        self.u_inf = u_inf  # Free stream velocity
        self.nu = nu        # Kinematic viscosity
    
    def get_reference_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Get the Blasius boundary layer solution (approximate, safe for x >= 0)."""
        x = np.array(x)
        # Avoid division by zero and negative roots
        x_safe = np.where(x <= 0, 1e-8, x)
        eta = x_safe * np.sqrt(self.u_inf / (2 * self.nu * x_safe))
        # Approximate solution for the Blasius equation
        f = 0.332 * eta
        f_prime = np.full_like(x_safe, 0.332)  # constant for all x
        f_double_prime = 0.332 * np.exp(-eta**2/2)
        u = self.u_inf * f_prime
        v = 0.5 * np.sqrt(self.nu * self.u_inf / x_safe) * (eta * f_prime - f)
        # For x <= 0, set u and v to 0
        u = np.where(x <= 0, 0.0, u)
        v = np.where(x <= 0, 0.0, v)
        return {
            'u': u,
            'v': v
        }

class TextbookCaseManager:
    """Manager class for handling multiple textbook cases."""
    def __init__(self):
        self.cases = {}
    
    def add_case(self, case: TextbookCase):
        """Add a textbook case to the manager."""
        self.cases[case.name] = case
    
    def validate_all(self,
                    cfd_data: Dict[str, Dict[str, np.ndarray]],
                    x_coords: Dict[str, np.ndarray],
                    save_dir: str) -> Dict[str, Dict[str, ValidationMetrics]]:
        """Validate CFD results against all textbook cases."""
        results = {}
        
        for case_name, case in self.cases.items():
            if case_name not in cfd_data:
                print(f"Warning: No CFD data found for {case_name}")
                continue
            
            case_save_dir = str(Path(save_dir) / case_name)
            results[case_name] = case.validate(
                cfd_data[case_name],
                x_coords[case_name],
                save_dir=case_save_dir
            )
        
        return results
    
    def generate_summary_report(self,
                              results: Dict[str, Dict[str, ValidationMetrics]],
                              save_dir: str) -> str:
        """Generate a summary report for all textbook cases."""
        report = []
        report.append("Textbook Case Validation Summary")
        report.append("=" * 50)
        
        for case_name, case_metrics in results.items():
            report.append(f"\nCase: {case_name}")
            report.append("-" * 30)
            
            for var_name, metrics in case_metrics.items():
                report.append(f"\nVariable: {var_name}")
                for key, value in metrics.__dict__.items():
                    if isinstance(value, tuple):
                        report.append(f"{key}: {value[0]:.4f} to {value[1]:.4f}")
                    else:
                        report.append(f"{key}: {value:.4f}")
        
        # Save report
        report_path = Path(save_dir) / "textbook_validation_summary.txt"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        return "\n".join(report) 