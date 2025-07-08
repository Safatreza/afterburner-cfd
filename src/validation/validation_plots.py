import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
from .validation_metrics import ValidationMetrics

class ValidationPlotter:
    def __init__(self, style: str = 'default'):
        """Initialize the plotter with a specific style."""
        plt.style.use(style)
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_comparison(self,
                       cfd_data: np.ndarray,
                       reference_data: np.ndarray,
                       x_data: Optional[np.ndarray] = None,
                       variable_name: str = "Variable",
                       x_label: str = "Position",
                       y_label: Optional[str] = None,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None):
        """Plot CFD results against reference data."""
        plt.figure(figsize=(10, 6))
        
        if x_data is None:
            x_data = np.arange(len(cfd_data))
        
        plt.plot(x_data, cfd_data, 'o-', label='CFD Results', color=self.colors[0])
        plt.plot(x_data, reference_data, 's--', label='Reference Data', color=self.colors[1])
        
        plt.xlabel(x_label)
        plt.ylabel(y_label or variable_name)
        plt.title(title or f"{variable_name} Comparison")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_error_distribution(self,
                              cfd_data: np.ndarray,
                              reference_data: np.ndarray,
                              variable_name: str = "Variable",
                              save_path: Optional[str] = None):
        """Plot error distribution between CFD and reference data."""
        errors = cfd_data - reference_data
        
        plt.figure(figsize=(10, 6))
        
        # Histogram of errors
        sns.histplot(errors, kde=True, color=self.colors[0])
        
        # Add vertical line at mean
        mean_error = np.mean(errors)
        plt.axvline(mean_error, color='red', linestyle='--', 
                   label=f'Mean Error: {mean_error:.4f}')
        
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution for {variable_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_grid_convergence(self,
                            fine_data: np.ndarray,
                            medium_data: np.ndarray,
                            coarse_data: np.ndarray,
                            grid_sizes: List[float],
                            variable_name: str = "Variable",
                            save_path: Optional[str] = None):
        """Plot grid convergence study results."""
        plt.figure(figsize=(10, 6))
        
        # Plot data points
        plt.loglog(grid_sizes, [np.mean(fine_data), np.mean(medium_data), np.mean(coarse_data)],
                  'o-', label='CFD Results', color=self.colors[0])
        
        # Add trend line
        z = np.polyfit(np.log10(grid_sizes), 
                      np.log10([np.mean(fine_data), np.mean(medium_data), np.mean(coarse_data)]), 1)
        p = np.poly1d(z)
        plt.loglog(grid_sizes, 10**p(np.log10(grid_sizes)), '--', 
                  label=f'Trend (p={z[0]:.2f})', color=self.colors[1])
        
        plt.xlabel('Grid Size')
        plt.ylabel(variable_name)
        plt.title(f'Grid Convergence Study for {variable_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_summary(self,
                           metrics: Dict[str, ValidationMetrics],
                           save_path: Optional[str] = None):
        """Create a summary plot of validation metrics."""
        # Prepare data for plotting
        variables = list(metrics.keys())
        metric_names = ['mae', 'rmse', 'relative_error']
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(variables))
        width = 0.25
        
        for i, metric in enumerate(metric_names):
            values = [getattr(metrics[var], metric) for var in variables]
            plt.bar(x + i*width, values, width, label=metric.upper())
        
        plt.xlabel('Variables')
        plt.ylabel('Error Metrics')
        plt.title('Validation Metrics Summary')
        plt.xticks(x + width, variables, rotation=45)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_uncertainty_bounds(self,
                              cfd_data: np.ndarray,
                              reference_data: np.ndarray,
                              x_data: Optional[np.ndarray] = None,
                              variable_name: str = "Variable",
                              confidence_level: float = 0.95,
                              save_path: Optional[str] = None):
        """Plot CFD results with uncertainty bounds."""
        plt.figure(figsize=(10, 6))
        
        if x_data is None:
            x_data = np.arange(len(cfd_data))
        
        # Calculate uncertainty bounds
        std = np.std(cfd_data - reference_data)
        ci = stats.t.interval(confidence_level, len(cfd_data)-1, 
                            loc=cfd_data, scale=std/np.sqrt(len(cfd_data)))
        
        # Plot data and uncertainty bounds
        plt.plot(x_data, cfd_data, 'o-', label='CFD Results', color=self.colors[0])
        plt.plot(x_data, reference_data, 's--', label='Reference Data', color=self.colors[1])
        plt.fill_between(x_data, ci[0], ci[1], alpha=0.2, 
                        label=f'{confidence_level*100}% Confidence Interval')
        
        plt.xlabel('Position')
        plt.ylabel(variable_name)
        plt.title(f'{variable_name} with Uncertainty Bounds')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_validation_report(self,
                               cfd_data: Dict[str, np.ndarray],
                               reference_data: Dict[str, np.ndarray],
                               metrics: Dict[str, ValidationMetrics],
                               save_dir: str):
        """Create a comprehensive validation report with all plots."""
        for var_name in cfd_data.keys():
            # Create comparison plot
            self.plot_comparison(
                cfd_data[var_name],
                reference_data[var_name],
                variable_name=var_name,
                save_path=f"{save_dir}/{var_name}_comparison.png"
            )
            
            # Create error distribution plot
            self.plot_error_distribution(
                cfd_data[var_name],
                reference_data[var_name],
                variable_name=var_name,
                save_path=f"{save_dir}/{var_name}_error_dist.png"
            )
            
            # Create uncertainty bounds plot
            self.plot_uncertainty_bounds(
                cfd_data[var_name],
                reference_data[var_name],
                variable_name=var_name,
                save_path=f"{save_dir}/{var_name}_uncertainty.png"
            )
        
        # Create metrics summary plot
        self.plot_metrics_summary(
            metrics,
            save_path=f"{save_dir}/metrics_summary.png"
        ) 