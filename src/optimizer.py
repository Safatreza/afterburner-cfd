import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import Dict, List, Any, Callable, Tuple
import json
from pathlib import Path
import logging
from parameter_study import ParameterStudy
from scipy.stats import norm

class AfterburnerOptimizer:
    def __init__(self, 
                 base_case_dir: str,
                 objective_type: str = "thrust",
                 optimization_method: str = "bayesian",
                 n_initial_points: int = 5):
        """Initialize the afterburner optimizer.
        
        Args:
            base_case_dir: Path to the base OpenFOAM case directory
            objective_type: Type of objective to optimize ("thrust" or "efficiency")
            optimization_method: Optimization method ("bayesian" or "scipy")
            n_initial_points: Number of initial points for Bayesian optimization
        """
        self.base_case_dir = Path(base_case_dir)
        self.objective_type = objective_type
        self.optimization_method = optimization_method
        self.n_initial_points = n_initial_points
        
        # Parameter bounds
        self.param_bounds = {
            "injectorAngle": (30.0, 60.0),  # degrees
            "injectorDiameter": (0.005, 0.02),  # meters
            "vGutterWidth": (0.02, 0.08),  # meters
            "inletVelocity": (50.0, 200.0),  # m/s
            "inletTemperature": (250.0, 400.0),  # K
            "turbulenceIntensity": (0.01, 0.1)  # dimensionless
        }
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization history
        self.history = {
            "parameters": [],
            "objectives": [],
            "constraints": []
        }
    
    def _create_parameter_study(self, params: Dict[str, float]) -> ParameterStudy:
        """Create a parameter study instance with the given parameters."""
        study_name = f"optimization_{self.objective_type}_{len(self.history['parameters'])}"
        return ParameterStudy(
            base_case_dir=str(self.base_case_dir),
            study_name=study_name
        )
    
    def _extract_objective(self, case_dir: Path) -> float:
        """Extract the objective value from simulation results.
        
        Args:
            case_dir: Path to the case directory
            
        Returns:
            float: Objective value (thrust or efficiency)
        """
        if self.objective_type == "thrust":
            # Read thrust from post-processing results
            thrust_file = case_dir / "postProcessing" / "forces" / "0" / "force.dat"
            if thrust_file.exists():
                with open(thrust_file, 'r') as f:
                    # Skip header and read last line
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split()
                        return float(last_line[1])  # Assuming thrust is in second column
        else:  # efficiency
            # Read efficiency from post-processing results
            efficiency_file = case_dir / "postProcessing" / "efficiency" / "0" / "efficiency.dat"
            if efficiency_file.exists():
                with open(efficiency_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split()
                        return float(last_line[1])
        
        return 0.0  # Default value if results not found
    
    def _check_constraints(self, case_dir: Path) -> List[float]:
        """Check if the simulation results satisfy constraints.
        
        Args:
            case_dir: Path to the case directory
            
        Returns:
            List[float]: Constraint violations (negative values indicate violations)
        """
        constraints = []
        
        # Check pressure limits
        pressure_file = case_dir / "postProcessing" / "pressure" / "0" / "pressure.dat"
        if pressure_file.exists():
            with open(pressure_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    pressures = [float(line.strip().split()[1]) for line in lines[1:]]
                    max_pressure = max(pressures)
                    constraints.append(1e6 - max_pressure)  # Max pressure constraint
        
        # Check temperature limits
        temperature_file = case_dir / "postProcessing" / "temperature" / "0" / "temperature.dat"
        if temperature_file.exists():
            with open(temperature_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    temperatures = [float(line.strip().split()[1]) for line in lines[1:]]
                    max_temp = max(temperatures)
                    constraints.append(2000 - max_temp)  # Max temperature constraint
        
        return constraints
    
    def _objective_function(self, x: np.ndarray) -> float:
        """Objective function for optimization.
        
        Args:
            x: Array of parameter values
            
        Returns:
            float: Negative objective value (for minimization)
        """
        # Convert array to parameter dictionary
        params = {
            "injectorAngle": x[0],
            "injectorDiameter": x[1],
            "vGutterWidth": x[2],
            "inletVelocity": x[3],
            "inletTemperature": x[4],
            "turbulenceIntensity": x[5]
        }
        
        # Create and run parameter study
        study = self._create_parameter_study(params)
        study.run_parameter_study([params])
        
        # Get objective value
        case_dir = study.study_dir / "case_000"
        objective = self._extract_objective(case_dir)
        
        # Check constraints
        constraints = self._check_constraints(case_dir)
        if any(c < 0 for c in constraints):
            return 1e6  # Penalty for constraint violation
        
        # Store in history
        self.history["parameters"].append(params)
        self.history["objectives"].append(objective)
        self.history["constraints"].append(constraints)
        
        return -objective  # Negative for minimization
    
    def _bayesian_optimization(self, n_iterations: int) -> Tuple[Dict[str, float], float]:
        """Perform Bayesian optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Tuple[Dict[str, float], float]: Best parameters and objective value
        """
        # Initialize Gaussian Process
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel)
        
        # Generate initial points
        X = []
        y = []
        for _ in range(self.n_initial_points):
            x = np.array([np.random.uniform(low, high) 
                         for low, high in self.param_bounds.values()])
            X.append(x)
            y.append(self._objective_function(x))
        
        X = np.array(X)
        y = np.array(y)
        
        # Optimization loop
        for i in range(n_iterations):
            # Update GP
            gp.fit(X, y)
            
            # Find next point to evaluate
            x_next = self._acquisition_function(gp, X)
            y_next = self._objective_function(x_next)
            
            X = np.vstack((X, x_next))
            y = np.append(y, y_next)
            
            self.logger.info(f"Iteration {i+1}: Objective = {-y_next}")
        
        # Find best parameters
        best_idx = np.argmin(y)
        best_params = {
            name: X[best_idx, i]
            for i, name in enumerate(self.param_bounds.keys())
        }
        
        return best_params, -y[best_idx]
    
    def _acquisition_function(self, gp: GaussianProcessRegressor, X: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function.
        
        Args:
            gp: Fitted Gaussian Process
            X: Previously evaluated points
            
        Returns:
            np.ndarray: Next point to evaluate
        """
        # Generate random points
        n_samples = 1000
        X_samples = np.array([
            np.random.uniform(low, high, n_samples)
            for low, high in self.param_bounds.values()
        ]).T
        
        # Predict mean and std
        y_mean, y_std = gp.predict(X_samples, return_std=True)
        
        # Calculate expected improvement
        y_best = np.min(gp.predict(X))
        z = (y_best - y_mean) / y_std
        ei = (y_best - y_mean) * norm.cdf(z) + y_std * norm.pdf(z)
        
        return X_samples[np.argmax(ei)]
    
    def optimize(self, n_iterations: int = 20) -> Tuple[Dict[str, float], float]:
        """Run the optimization.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Tuple[Dict[str, float], float]: Best parameters and objective value
        """
        if self.optimization_method == "bayesian":
            return self._bayesian_optimization(n_iterations)
        else:  # scipy
            # Define bounds for scipy optimizers
            bounds = list(self.param_bounds.values())
            
            # Try different optimization methods
            methods = ["differential_evolution", "L-BFGS-B"]
            best_result = None
            best_objective = float('inf')
            
            for method in methods:
                try:
                    if method == "differential_evolution":
                        result = differential_evolution(
                            self._objective_function,
                            bounds=bounds,
                            maxiter=n_iterations,
                            popsize=10
                        )
                    else:
                        result = minimize(
                            self._objective_function,
                            x0=np.array([np.mean(b) for b in bounds]),
                            bounds=bounds,
                            method=method,
                            options={'maxiter': n_iterations}
                        )
                    
                    if result.fun < best_objective:
                        best_result = result
                        best_objective = result.fun
                
                except Exception as e:
                    self.logger.warning(f"Optimization with {method} failed: {str(e)}")
            
            if best_result is None:
                raise RuntimeError("All optimization methods failed")
            
            # Convert result to parameter dictionary
            best_params = {
                name: best_result.x[i]
                for i, name in enumerate(self.param_bounds.keys())
            }
            
            return best_params, -best_objective
    
    def save_results(self, output_dir: str):
        """Save optimization results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save history
        with open(output_dir / "optimization_history.json", 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Save best parameters
        best_idx = np.argmax(self.history["objectives"])
        best_params = self.history["parameters"][best_idx]
        best_objective = self.history["objectives"][best_idx]
        
        with open(output_dir / "best_parameters.json", 'w') as f:
            json.dump({
                "parameters": best_params,
                "objective": best_objective
            }, f, indent=4) 