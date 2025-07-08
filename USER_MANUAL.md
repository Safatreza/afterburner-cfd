# User Manual

## Project Summary
This project simulates compressible flow in a supersonic afterburner using Rayleigh flow, with modules for heat addition, shock detection, validation, and visualization. It is designed for research and educational use.

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/afterburner-cfd.git
   cd afterburner-cfd
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Full Version
Run the main simulation with custom parameters:
```bash
python src/main.py --mach 2.0 --pressure 101325 --temperature 300 --heat_peak 1e6 --heat_width 0.1 --length 1.0 --points 1000 --export --output results.csv
```

## How to Run the Demo Version
Run the minimal, dependency-free demo:
```bash
python demo/demo.py
```

## How to Run Tests
Run all unit tests using pytest:
```bash
pytest tests/
```

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