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
