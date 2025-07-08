# User Manual

## Project Summary
This project simulates compressible flow in a supersonic afterburner using Rayleigh flow, with modules for heat addition, shock detection, validation, and visualization. It is designed for research and educational use.

## Installation Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Safatreza/afterburner-cfd.git
   cd afterburner-cfd
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Full Version
Run the main simulation with custom parameters:
```bash
python src/main.py --mach 2.0 --pressure 101325 --temperature 300 --heat_peak 1e6 --heat_width 0.1 --length 1.0 --points 1000 --export --output results.csv
```
- This will run the full CFD simulation and output results to a CSV file.

## How to Run the Demo Version
Run the minimal, dependency-free demo:
```bash
python demo/demo.py
```
- This prints and plots a simple Mach number profile for educational purposes.

## How to Run Tests
Run all unit tests using pytest:
```bash
pytest tests/
```
- This will check the core solver, shock detection, heat profile, and validation metrics.

## How to Use Docker
Build and run the simulation in a container:
```bash
docker build -t afterburner-cfd .
docker run --rm afterburner-cfd
```
Or use docker-compose:
```bash
docker-compose up --build
```

## Troubleshooting
- **Matplotlib plot does not show:** Ensure you are running in a local environment with a GUI.
- **Gmsh not found:** Install Gmsh or open the `.msh` file manually in Gmsh.
- **Git errors:** Make sure you are in the correct directory and have initialized git.
- **Dependency issues:** Double-check your Python version and run `pip install -r requirements.txt` again.

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

## Additional Resources
- See `README.md` for a quick overview.
- See `PROJECT_SUMMARY.md` for a detailed summary and code structure explanation. 