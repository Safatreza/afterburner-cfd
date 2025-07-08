# Supersonic Afterburner CFD Simulation

A modular Python tool for simulating compressible flow in a supersonic afterburner, featuring heat addition, shock detection, validation, and visualization. Designed for research and education.

---

## Features
- Supersonic inlet/outlet boundary conditions
- Configurable spatial heat addition
- Shock detection and visualization
- Validation against experimental/textbook data
- Real-time and static plots
- Command-line and Python API
- Modular, extensible codebase
- CI/CD with GitHub Actions
- Docker support for easy deployment

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
### Full Simulation
Run with custom parameters:
```bash
python src/main.py --mach 2.0 --pressure 101325 --temperature 300 --heat_peak 1e6 --heat_width 0.1 --length 1.0 --points 1000 --export --output results.csv
```

### Demo Version
Run a minimal, dependency-free demo:
```bash
python demo/demo.py
```

### Run Tests
```bash
pytest tests/
```

---

## DevOps & CI/CD
- Automated tests and linting via GitHub Actions ([see workflow](.github/workflows/python-ci.yml))
- Test coverage report generated on each push

---

## Docker Support
Build and run the simulation in a container:
```bash
docker build -t afterburner-cfd .
docker run --rm afterburner-cfd
```
Or use docker-compose:
```bash
docker-compose up --build
```

---

## Folder Structure
```
afterburner-cfd/
├── src/                # Core logic
│   └── validation/     # Validation modules
├── demo/               # Minimal demo version
├── tests/              # Unit tests
├── docs/               # Documentation
├── test_data/          # Experimental/reference data
├── test_results/       # Output plots, reports, CSVs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── USER_MANUAL.md
└── ...
```

---

## Documentation
See [USER_MANUAL.md](USER_MANUAL.md) for detailed instructions, workflow, and troubleshooting.
