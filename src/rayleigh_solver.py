import numpy as np
from scipy.integrate import solve_ivp
<<<<<<< HEAD
from typing import Dict, Any, Tuple
=======
from typing import Dict
>>>>>>> 5b8c4d6 (Final project update: code cleanup, visualization, docs, and tests)

class RayleighFlowSolver:
    def __init__(self, config):
        """Initialize the Rayleigh flow solver with configuration parameters."""
        self.config = config
        self.gamma = 1.4  # Specific heat ratio for air
        self.R = 287.0    # Gas constant for air (J/kg·K)
        
        # Initialize grid
        self.x = np.linspace(0, config.length, config.points)
        self.dx = self.x[1] - self.x[0]
        
        # Initialize flow properties
        self.mach = np.zeros_like(self.x)
        self.pressure = np.zeros_like(self.x)
        self.temperature = np.zeros_like(self.x)
        self.density = np.zeros_like(self.x)
        self.velocity = np.zeros_like(self.x)
        
        # Set inlet conditions
        self._set_inlet_conditions()
    
    def _set_inlet_conditions(self):
        """Set the inlet conditions for the flow."""
        self.mach[0] = self.config.mach
        self.pressure[0] = self.config.pressure
        self.temperature[0] = self.config.temperature
        self.density[0] = self.pressure[0] / (self.R * self.temperature[0])
        self.velocity[0] = self.mach[0] * np.sqrt(self.gamma * self.R * self.temperature[0])
    
    def _heat_addition_profile(self, x: float) -> float:
        """Calculate the heat addition profile using a Gaussian distribution."""
        x_peak = self.config.length / 2
        return self.config.heat_peak * np.exp(-(x - x_peak)**2 / (2 * self.config.heat_width**2))
    
    def _flow_equations(self, x: float, y: np.ndarray) -> np.ndarray:
        """Define the system of ODEs for Rayleigh flow with heat addition."""
        M, p, T = y
        
        # Calculate heat addition at current position
        q = self._heat_addition_profile(x)
        
        # Calculate derivatives
        dM_dx = (M * (1 + self.gamma * M**2) * q) / (2 * p * np.sqrt(self.gamma * self.R * T))
        dp_dx = -self.gamma * M**2 * dM_dx * p / M
        dT_dx = (q - self.gamma * self.R * T * dM_dx * M) / (self.gamma * self.R)
        
        return np.array([dM_dx, dp_dx, dT_dx])
    
    def solve(self) -> Dict[str, np.ndarray]:
        """Solve the Rayleigh flow equations and return the results."""
        # Initial conditions
        y0 = np.array([self.mach[0], self.pressure[0], self.temperature[0]])
        
        # Solve ODE system
        solution = solve_ivp(
            self._flow_equations,
            (0, self.config.length),
            y0,
            method='RK45',
            t_eval=self.x,
            rtol=1e-8,
            atol=1e-8
        )
        
        # Extract results
        self.mach = solution.y[0]
        self.pressure = solution.y[1]
        self.temperature = solution.y[2]
        
        # Calculate derived quantities
        self.density = self.pressure / (self.R * self.temperature)
        self.velocity = self.mach * np.sqrt(self.gamma * self.R * self.temperature)
        
        # Detect shocks
        shock_locations = self._detect_shocks()
        
        return {
            'x': self.x,
            'mach': self.mach,
            'pressure': self.pressure,
            'temperature': self.temperature,
            'density': self.density,
            'velocity': self.velocity,
            'shock_locations': shock_locations
        }
    
    def _detect_shocks(self) -> np.ndarray:
        """Detect shock locations based on pressure gradient."""
        dp_dx = np.gradient(self.pressure, self.x)
        shock_threshold = 0.1 * np.max(np.abs(dp_dx))
        shock_locations = self.x[np.abs(dp_dx) > shock_threshold]
        return shock_locations
