# Supersonic Afterburner CFD Simulation

A Python-based simulation tool for modeling compressible flow in a quasi-1D afterburner using Rayleigh flow with spatial heat addition. This project is designed for research and educational purposes, enabling the study of supersonic combustion, heat addition, and shock phenomena in afterburner configurations. The solver features a Gaussian heat injection model, robust shock detection, real-time visualization, and a comprehensive validation module for comparison with experimental and textbook data.

---

## Features
- **Supersonic Inlet/Outlet Boundary Conditions**: Simulate realistic afterburner entry and exit conditions.
- **Spatial Heat Addition**: Model heat release using a configurable Gaussian profile.
- **Shock Detection**: Automatically identify and locate shocks in the flow field.
- **Validation Module**: Compare simulation results with experimental and textbook cases using quantitative error metrics.
- **Real-Time Visualization**: Generate static and animated plots of key flow variables (Mach number, pressure, temperature, density).
- **Command-Line Interface (CLI)**: Run simulations and export results directly from the terminal.
- **CSV Export**: Save simulation results for further analysis.
- **Interactive Animation**: Visualize the evolution of flow properties along the afterburner.

---

## Validation Module
The validation module ensures the accuracy and reliability of the simulation by:
- **Experimental Comparison**: Validates against published experimental data (see `test_data/experimental_data.json`).
- **Textbook Cases**: Benchmarks against classic Rayleigh flow and other analytical solutions.
- **Error Metrics**:
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Square Error)
  - **R²** (Coefficient of Determination)
- **Automated Reporting**: Generates summary plots and text reports in `test_results/` for both experimental and textbook validations.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/afterburner-cfd.git
   cd afterburner-cfd
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
### Command-Line Interface
Run a simulation with a custom configuration:
```bash
python src/main.py --mach 2.0 --pressure 101325 --temperature 300 --heat_peak 1e6 --heat_width 0.1 --length 1.0 --points 1000 --export --output results.csv
```

### Python API Example
```python
from src.rayleigh_solver import RayleighFlowSolver
config = {
    'mach': 2.0,
    'pressure': 101325,
    'temperature': 300,
    'heat_peak': 1e6,
    'heat_width': 0.1,
    'length': 1.0,
    'points': 1000
}
solver = RayleighFlowSolver(config)
results = solver.solve()
```

### Validation
Run the validation module:
```bash
python src/validation/test_validation.py
```

---

## Visualization
- **Static Plots**: Mach number, pressure, temperature, and density profiles along the afterburner.
- **Shock Location**: Highlighted in plots and exported as a separate file.
- **Animated Plots**: Real-time animation of flow evolution (requires Matplotlib).
- **Validation Plots**: Error distributions, comparison with experimental/textbook data, and uncertainty bands.

All plots and reports are saved in the `test_results/` directory.

---

## Project Structure
```
afterburner-cfd/
├── src/
│   ├── main.py                # CLI entry point
│   ├── rayleigh_solver.py     # Core Rayleigh flow solver
│   ├── plots.py               # Visualization utilities
│   ├── utils.py               # Config, export, and helper functions
│   └── validation/            # Validation and benchmarking modules
│       ├── experimental_comparison.py
│       ├── textbook_cases.py
│       ├── validation_metrics.py
│       ├── validation_plots.py
│       └── test_validation.py
├── test_data/                 # Experimental and reference data
├── test_results/              # Output plots, reports, and CSVs
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── ...
```

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository and create a new branch.
2. Make your changes with clear commit messages.
3. Ensure all tests pass and add new tests as needed.
4. Submit a pull request describing your changes.

For major changes, please open an issue first to discuss your proposal.

---

## License
[MIT License](LICENSE)  <!-- Replace with your actual license -->

---

## Acknowledgments
- Experimental data and textbook references as cited in the code and documentation.
- Inspired by classic CFD literature, including:
  - J. D. Anderson, *Computational Fluid Dynamics: The Basics with Applications*, McGraw-Hill, 1995.
  - F. R. Menter, "Two-Equation Eddy-Viscosity Turbulence Models for Engineering Applications," AIAA Journal, 32(8), 1994.
- Thanks to all contributors and the open-source scientific Python community.
