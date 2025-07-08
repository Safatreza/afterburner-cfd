# Project Summary: Supersonic Afterburner CFD Simulation

## Purpose
This project provides a modular Python-based tool for simulating compressible flow in a supersonic afterburner. It is designed for both research and educational use, enabling users to:
- Study Rayleigh flow with spatial heat addition
- Detect and analyze shock phenomena
- Validate results against experimental and textbook data
- Visualize flow properties and validation metrics

## Main Features
- **Supersonic Inlet/Outlet Boundary Conditions**
- **Configurable Heat Addition (Gaussian profile)**
- **Shock Detection**
- **Validation Module** (with MAE, RMSE, R² metrics)
- **Real-Time and Static Visualization**
- **Command-Line and Python API**
- **Docker and CI/CD Support**

## Code Structure
```
afterburner-cfd/
├── src/                # Core logic
│   ├── main.py         # CLI entry point
│   ├── rayleigh_solver.py     # Rayleigh flow solver
│   ├── compressible_ns_solver.py # 2D NS solver
│   ├── plots.py        # Visualization utilities
│   ├── utils.py        # Config, export, helpers
│   ├── gmsh_exporter.py# Gmsh mesh export
│   └── validation/     # Validation modules
│       ├── experimental_comparison.py
│       ├── textbook_cases.py
│       ├── validation_metrics.py
│       ├── validation_plots.py
│       └── test_validation.py
├── demo/               # Minimal, dependency-free demo
├── tests/              # Unit tests for core logic
├── docs/               # Documentation (user manual, report)
├── test_data/          # Experimental/reference data
├── test_results/       # Output plots, reports, CSVs
├── Dockerfile          # Containerization
├── docker-compose.yml  # Local dev container
├── requirements.txt    # Python dependencies
├── README.md           # Project overview
├── USER_MANUAL.md      # User manual
└── PROJECT_SUMMARY.md  # This file
```

## Key Modules Explained
- **main.py**: Entry point for running the full simulation via CLI.
- **rayleigh_solver.py**: Implements the Rayleigh flow solver for quasi-1D afterburner flow.
- **compressible_ns_solver.py**: 2D compressible Navier-Stokes solver (advanced usage).
- **plots.py**: Functions for plotting Mach number, pressure, temperature, density, and validation results.
- **utils.py**: Configuration management, results export, and helper functions.
- **gmsh_exporter.py**: Exports simulation results to Gmsh format for visualization.
- **validation/**: Contains modules for comparing simulation results to experimental/textbook data, computing error metrics, and generating validation plots.
- **demo/demo.py**: Minimal, dependency-free script to demonstrate Mach number profile evolution.
- **tests/**: Unit tests for solver, shock detection, heat profile, and validation metrics.

## Typical Workflow
1. **User provides input parameters** (via CLI or config)
2. **RayleighFlowSolver** computes the flow field
3. **Heat addition and shock detection** are applied
4. **Validation module** compares results to reference data
5. **Visualization and export** of results and validation metrics

## Who Should Use This?
- Researchers studying supersonic combustion or afterburner flows
- Students learning about compressible flow, Rayleigh flow, or CFD
- Engineers needing a modular, extensible CFD code for rapid prototyping

## Getting Started
- See `USER_MANUAL.md` for step-by-step instructions
- Run the demo for a quick, dependency-free example
- Use the full simulation for advanced analysis and validation

## Workflow Diagram
```
[User Input]
     ↓
[RayleighFlowSolver]
     ↓
[Heat Addition + Shock Detection]
     ↓
[Validation Module] ← [Experimental/Textbook Data]
     ↓
[Visualization + Export]
``` 