import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import json

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    r2: float   # R-squared
    max_error: float  # Maximum Error
    mean_error: float  # Mean Error
    std_error: float  # Standard Deviation of Error
    relative_error: float  # Relative Error (%)
    confidence_interval: Tuple[float, float]  # 95% Confidence Interval

class ValidationAnalyzer:
    def __init__(self):
        self.metrics_history = []
    
    def compute_metrics(self, 
                       cfd_data: np.ndarray, 
                       reference_data: np.ndarray,
                       variable_name: str = "unknown") -> ValidationMetrics:
        """Compute validation metrics between CFD and reference data."""
        # Ensure data is numpy array
        cfd_data = np.array(cfd_data)
        reference_data = np.array(reference_data)
        
        # Compute errors
        errors = cfd_data - reference_data
        abs_errors = np.abs(errors)
        
        # Compute metrics
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors**2))
        r2 = 1 - np.sum(errors**2) / np.sum((reference_data - np.mean(reference_data))**2)
        max_error = np.max(abs_errors)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Compute relative error, excluding zero reference values
        nonzero_mask = np.abs(reference_data) > 1e-12
        if np.any(nonzero_mask):
            relative_error = np.mean(abs_errors[nonzero_mask] / np.abs(reference_data[nonzero_mask])) * 100
            num_excluded = np.sum(~nonzero_mask)
        else:
            relative_error = float('nan')
            num_excluded = len(reference_data)
        
        # Compute 95% confidence interval
        confidence_interval = stats.t.interval(
            0.95,
            len(errors)-1,
            loc=np.mean(errors),
            scale=stats.sem(errors)
        )
        
        metrics = ValidationMetrics(
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            max_error=float(max_error),
            mean_error=float(mean_error),
            std_error=float(std_error),
            relative_error=float(relative_error),
            confidence_interval=tuple(map(float, confidence_interval))
        )
        
        # Store metrics in history, including number of excluded points
        self.metrics_history.append({
            'variable': variable_name,
            'metrics': metrics.__dict__,
            'excluded_points_for_relative_error': int(num_excluded)
        })
        
        return metrics
    
    def compute_grid_convergence(self, 
                               fine_data: np.ndarray,
                               medium_data: np.ndarray,
                               coarse_data: np.ndarray,
                               fine_grid_size: float,
                               medium_grid_size: float,
                               coarse_grid_size: float) -> Dict[str, float]:
        """Compute grid convergence metrics using Richardson extrapolation."""
        # Compute grid refinement ratios
        r21 = medium_grid_size / fine_grid_size
        r32 = coarse_grid_size / medium_grid_size
        
        # Compute convergence metrics
        p = np.log(np.abs(medium_data - coarse_data) / np.abs(fine_data - medium_data)) / np.log(r21)
        p = np.mean(p)  # Average order of accuracy
        
        # Compute grid convergence index (GCI)
        Fs = 1.25  # Safety factor
        gci_fine = Fs * np.abs(fine_data - medium_data) / (r21**p - 1)
        gci_medium = Fs * np.abs(medium_data - coarse_data) / (r32**p - 1)
        
        # Check for asymptotic range
        asymptotic_range = gci_medium / (r21**p * gci_fine)
        
        return {
            'order_of_accuracy': float(p),
            'gci_fine': float(np.mean(gci_fine)),
            'gci_medium': float(np.mean(gci_medium)),
            'asymptotic_range': float(asymptotic_range)
        }
    
    def compute_uncertainty(self, 
                          data: np.ndarray,
                          confidence_level: float = 0.95) -> Dict[str, float]:
        """Compute uncertainty metrics for the data."""
        # Compute basic statistics
        mean = np.mean(data)
        std = np.std(data)
        
        # Compute confidence interval
        ci = stats.t.interval(
            confidence_level,
            len(data)-1,
            loc=mean,
            scale=stats.sem(data)
        )
        
        # Compute expanded uncertainty (k=2 for 95% confidence)
        expanded_uncertainty = 2 * std
        
        return {
            'mean': float(mean),
            'std': float(std),
            'confidence_interval': tuple(map(float, ci)),
            'expanded_uncertainty': float(expanded_uncertainty)
        }
    
    def save_metrics(self, filepath: str):
        """Save validation metrics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    @classmethod
    def load_metrics(cls, filepath: str) -> 'ValidationAnalyzer':
        """Load validation metrics from file."""
        analyzer = cls()
        with open(filepath, 'r') as f:
            analyzer.metrics_history = json.load(f)
        return analyzer
    
    def generate_report(self) -> str:
        """Generate a validation report."""
        report = []
        report.append("Validation Report")
        report.append("=" * 50)
        
        for entry in self.metrics_history:
            report.append(f"\nVariable: {entry['variable']}")
            report.append("-" * 30)
            metrics = entry['metrics']
            for key, value in metrics.items():
                if isinstance(value, tuple):
                    report.append(f"{key}: {value[0]:.4f} to {value[1]:.4f}")
                else:
                    report.append(f"{key}: {value:.4f}")
            if 'excluded_points_for_relative_error' in entry:
                report.append(f"Excluded points for relative error: {entry['excluded_points_for_relative_error']}")
        
        return "\n".join(report) 