import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import json
from .validation_metrics import ValidationMetrics, ValidationAnalyzer
from .validation_plots import ValidationPlotter

class ExperimentalComparison:
    def __init__(self, experimental_data_path: Optional[str] = None):
        """Initialize the experimental comparison module."""
        self.experimental_data = {}
        self.uncertainty_data = {}
        self.analyzer = ValidationAnalyzer()
        self.plotter = ValidationPlotter()
        
        if experimental_data_path:
            self.load_experimental_data(experimental_data_path)
    
    def load_experimental_data(self, data_path: str):
        """Load experimental data from file."""
        data_path = Path(data_path)
        
        if data_path.suffix == '.csv':
            # Load from CSV
            df = pd.read_csv(data_path)
            for column in df.columns:
                if column.endswith('_uncertainty'):
                    var_name = column.replace('_uncertainty', '')
                    self.uncertainty_data[var_name] = df[column].values
                else:
                    self.experimental_data[column] = df[column].values
        
        elif data_path.suffix == '.json':
            # Load from JSON
            with open(data_path, 'r') as f:
                data = json.load(f)
                self.experimental_data = data.get('data', {})
                self.uncertainty_data = data.get('uncertainty', {})
        
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    def compare_with_cfd(self,
                        cfd_data: Dict[str, np.ndarray],
                        variables: Optional[List[str]] = None,
                        save_dir: Optional[str] = None) -> Dict[str, ValidationMetrics]:
        """Compare CFD results with experimental data."""
        if variables is None:
            variables = list(cfd_data.keys())
        
        metrics = {}
        
        for var_name in variables:
            if var_name not in self.experimental_data:
                print(f"Warning: No experimental data found for {var_name}")
                continue
            
            # Get experimental data and uncertainty
            exp_data = self.experimental_data[var_name]
            exp_uncertainty = self.uncertainty_data.get(var_name, None)
            
            # Compute metrics
            metrics[var_name] = self.analyzer.compute_metrics(
                cfd_data[var_name],
                exp_data,
                variable_name=var_name
            )
            
            # Create comparison plots if save_dir is provided
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                
                # Plot comparison
                self.plotter.plot_comparison(
                    cfd_data[var_name],
                    exp_data,
                    variable_name=var_name,
                    save_path=str(save_dir_path / f"{var_name}_comparison.png")
                )
                
                # Plot error distribution
                self.plotter.plot_error_distribution(
                    cfd_data[var_name],
                    exp_data,
                    variable_name=var_name,
                    save_path=str(save_dir_path / f"{var_name}_error_dist.png")
                )
                
                # Plot with uncertainty bounds if available
                if exp_uncertainty is not None:
                    self.plotter.plot_uncertainty_bounds(
                        cfd_data[var_name],
                        exp_data,
                        variable_name=var_name,
                        save_path=str(save_dir_path / f"{var_name}_uncertainty.png")
                    )
        
        # Create summary plot if save_dir is provided
        if save_dir:
            self.plotter.plot_metrics_summary(
                metrics,
                save_path=str(Path(save_dir) / "metrics_summary.png")
            )
        
        return metrics
    
    def compute_validation_ratio(self,
                               cfd_data: Dict[str, np.ndarray],
                               variables: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute validation ratio (VR) for each variable."""
        if variables is None:
            variables = list(cfd_data.keys())
        
        vr_dict = {}
        
        for var_name in variables:
            if var_name not in self.experimental_data:
                continue
            
            exp_data = self.experimental_data[var_name]
            exp_uncertainty = self.uncertainty_data.get(var_name, None)
            
            if exp_uncertainty is None:
                print(f"Warning: No uncertainty data found for {var_name}")
                continue
            
            # Compute validation ratio
            error = np.abs(cfd_data[var_name] - exp_data)
            vr = error / exp_uncertainty
            vr_dict[var_name] = float(np.mean(vr))
        
        return vr_dict
    
    def generate_validation_report(self,
                                 cfd_data: Dict[str, np.ndarray],
                                 save_dir: str,
                                 variables: Optional[List[str]] = None) -> str:
        """Generate a comprehensive validation report."""
        # Compute metrics
        metrics = self.compare_with_cfd(cfd_data, variables, save_dir)
        
        # Compute validation ratios
        vr_dict = self.compute_validation_ratio(cfd_data, variables)
        
        # Generate report
        report = []
        report.append("Validation Report")
        report.append("=" * 50)
        
        for var_name in metrics.keys():
            report.append(f"\nVariable: {var_name}")
            report.append("-" * 30)
            
            # Add metrics
            for key, value in metrics[var_name].__dict__.items():
                if isinstance(value, tuple):
                    report.append(f"{key}: {value[0]:.4f} to {value[1]:.4f}")
                else:
                    report.append(f"{key}: {value:.4f}")
            
            # Add validation ratio if available
            if var_name in vr_dict:
                report.append(f"Validation Ratio: {vr_dict[var_name]:.4f}")
        
        # Save report
        report_path = Path(save_dir) / "validation_report.txt"
        with open(report_path, 'w') as f:
            f.write("\n".join(report))
        
        return "\n".join(report)
    
    def save_experimental_data(self, save_path: str):
        """Save experimental data to file."""
        data_path = Path(save_path)
        
        if data_path.suffix == '.csv':
            # Save as CSV
            df = pd.DataFrame(self.experimental_data)
            for var_name, uncertainty in self.uncertainty_data.items():
                df[f"{var_name}_uncertainty"] = uncertainty
            df.to_csv(data_path, index=False)
        
        elif data_path.suffix == '.json':
            # Save as JSON
            data = {
                'data': self.experimental_data,
                'uncertainty': self.uncertainty_data
            }
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=4)
        
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.") 