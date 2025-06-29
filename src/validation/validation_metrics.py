import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Union
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

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

def monte_carlo_uncertainty(sim_func, base_config, n_samples=100, mach_range=(1.1,1.3), heat_range=(8e5,1.2e6)):
    results = []
    mach_samples = np.random.uniform(mach_range[0], mach_range[1], n_samples)
    heat_samples = np.random.uniform(heat_range[0], heat_range[1], n_samples)
    for m, q in zip(mach_samples, heat_samples):
        config = base_config.copy()
        config['mach'] = m
        config['heat_source'] = config.get('heat_source', {}).copy()
        config['heat_source']['amplitude'] = q
        results.append(sim_func(config))
    # Aggregate results (assume results is a list of dicts with arrays)
    keys = results[0].keys()
    agg = {k: np.array([r[k] for r in results]) for k in keys if isinstance(results[0][k], np.ndarray)}
    return agg

def plot_uncertainty_band(agg, var='mach', idx=0):
    # Plot mean and 95% band for a variable at a given y-index
    arr = agg[var][:, :, idx]  # shape: (n_samples, nx)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    x = agg['x'][0]
    plt.fill_between(x, mean-2*std, mean+2*std, alpha=0.3, label='95% band')
    plt.plot(x, mean, label='mean')
    plt.xlabel('x')
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def generate_pdf_report(metrics, plots, filename='validation_report.pdf'):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont('Helvetica', 14)
    c.drawString(30, height-40, 'Validation Report')
    c.setFont('Helvetica', 10)
    y = height-70
    for k, v in metrics.items():
        c.drawString(30, y, f'{k}: {v}')
        y -= 15
    y -= 10
    for plot_buf in plots:
        c.drawImage(plot_buf, 30, y-200, width=500, height=200)
        y -= 220
    c.save()

def compute_error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute error metrics between true and predicted arrays."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    max_error = np.max(np.abs(y_true - y_pred))
    mean_bias = np.mean(y_pred - y_true)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Max Error': max_error, 'Mean Bias': mean_bias}

def generate_batch_pdf_report(metrics_list: list, plots_list: list, filenames: list, out_pdf: str = 'batch_validation_report.pdf') -> None:
    """Generate a batch PDF report for multiple cases."""
    c = canvas.Canvas(out_pdf, pagesize=letter)
    width, height = letter
    c.setFont('Helvetica', 14)
    c.drawString(30, height-40, 'Batch Validation Report')
    y = height-70
    for i, (metrics, plots, fname) in enumerate(zip(metrics_list, plots_list, filenames)):
        c.setFont('Helvetica-Bold', 12)
        c.drawString(30, y, f'Case: {fname}')
        y -= 20
        c.setFont('Helvetica', 10)
        for k, v in metrics.items():
            c.drawString(30, y, f'{k}: {v}')
            y -= 15
        y -= 10
        for plot_buf in plots:
            c.drawImage(plot_buf, 30, y-200, width=500, height=200)
            y -= 220
        y -= 20
        if y < 250:
            c.showPage()
            y = height-70
    c.save() 