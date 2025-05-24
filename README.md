# Supersonic Afterburner CFD Simulation

This project simulates the flow dynamics in a **supersonic afterburner** using **Rayleigh flow** with spatial heat addition. The program implements a **quasi-1D CFD model** to solve compressible fluid flow equations, including a **Gaussian heat injection** model. It also provides visualizations of the flow properties (Mach number, pressure, temperature, etc.) and can detect shocks in the flow.

## 🚀 Features

- **Supersonic inlet flow** (Mach 1.0–1.5)
- **Heat injection model**: Gaussian spatial heat addition
- **Rayleigh flow solver**: Solving compressible flow equations with heat addition
- **Shock detection**: Detects sudden pressure drops or Mach number discontinuities
- **Real-time animation**: Displays the evolution of the Mach number over time
- **CLI parameter support**: Customize simulation parameters via command-line arguments
- **CSV Export**: Save final simulation results to a CSV file for further analysis
- **Visualization**: Plots the Mach number, pressure, temperature, density, and velocity profiles

## 📊 Validation Module

The project includes a comprehensive validation module that compares CFD results against experimental data and textbook cases. This ensures the accuracy and reliability of the simulation results.

### Features
- **Experimental Data Validation**: Compare CFD results with experimental measurements
  - Computes error metrics (MAE, RMSE, R², etc.)
  - Generates comparison plots and error distributions
  - Handles measurement uncertainties
  - Produces detailed validation reports

- **Textbook Case Validation**: Validate against analytical solutions
  - Laminar Poiseuille Flow
  - Blasius Boundary Layer
  - Customizable validation metrics
  - Automatic report generation

### Usage
```python
from src.validation.experimental_comparison import ExperimentalComparison
from src.validation.textbook_cases import TextbookCaseManager

# Validate against experimental data
comparison = ExperimentalComparison('path/to/experimental_data.json')
metrics = comparison.compare_with_cfd(cfd_data, save_dir='validation_results')

# Validate against textbook cases
manager = TextbookCaseManager()
manager.add_case(LaminarPoiseuilleFlow(dp_dx=-1.0, mu=1.0, rho=1.0, h=1.0))
results = manager.validate_all(cfd_data, x_coords, save_dir='validation_results')
```

### Dependencies
Additional dependencies for validation:
- `pandas`
- `seaborn`
- `scikit-learn`

Install validation-specific dependencies:
```bash
pip install -r src/validation/requirements.txt
```

## 🛠️ Requirements

- **Python 3.10+**
- **Dependencies**:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `tqdm`
  
Install the required libraries using the following command:

```bash
pip install -r requirements.txt
