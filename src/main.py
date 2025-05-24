import argparse
from pathlib import Path
from typing import Dict, Any
import subprocess

from rayleigh_solver import RayleighFlowSolver
from plots import FlowVisualizer
from utils import SimulationConfig, ResultsExporter
from gmsh_exporter import write_gmsh_2d

class AfterburnerSimulation:
    def __init__(self, config: Dict[str, Any]):
        self.config = SimulationConfig(config)
        self.solver = RayleighFlowSolver(self.config)
        self.visualizer = FlowVisualizer()
        self.exporter = ResultsExporter()
    
    def run(self):
        """Run the complete afterburner simulation."""
        print("Starting afterburner simulation...")
        
        # Solve the flow equations
        results = self.solver.solve()
        
        # Visualize results
        self.visualizer.plot_results(results)
        
        # Export results if requested
        if self.config.export_results:
            self.exporter.export_to_csv(results, self.config.output_path)
        
        # Export Gmsh visualization
        write_gmsh_2d(results, 'results.msh', height=2.0, ny=30)
        
        # Launch Gmsh to visualize the results
        try:
            subprocess.run(['gmsh', 'results.msh'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error launching Gmsh: {e}")
        except FileNotFoundError:
            print("Gmsh not found. Please open results.msh manually in Gmsh.")
        
        print("Simulation completed successfully!")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Supersonic Afterburner CFD Simulation")
    
    # Flow parameters
    parser.add_argument("--mach", type=float, default=1.2,
                      help="Inlet Mach number (default: 1.2)")
    parser.add_argument("--pressure", type=float, default=101325.0,
                      help="Inlet pressure in Pa (default: 101325.0)")
    parser.add_argument("--temperature", type=float, default=300.0,
                      help="Inlet temperature in K (default: 300.0)")
    
    # Heat addition parameters
    parser.add_argument("--heat_peak", type=float, default=1e6,
                      help="Peak heat addition in W/m³ (default: 1e6)")
    parser.add_argument("--heat_width", type=float, default=0.1,
                      help="Width of heat addition zone in m (default: 0.1)")
    
    # Simulation parameters
    parser.add_argument("--length", type=float, default=1.0,
                      help="Length of the afterburner in m (default: 1.0)")
    parser.add_argument("--points", type=int, default=1000,
                      help="Number of grid points (default: 1000)")
    
    # Output parameters
    parser.add_argument("--export", action="store_true",
                      help="Export results to CSV")
    parser.add_argument("--output", type=str, default="results.csv",
                      help="Output file path (default: results.csv)")
    
    return parser.parse_args()

def main():
    """Main entry point of the program."""
    args = parse_arguments()
    
    # Convert arguments to configuration dictionary
    config = vars(args)
    
    # Create and run simulation
    simulation = AfterburnerSimulation(config)
    simulation.run()

if __name__ == "__main__":
    main()
