import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
from matplotlib.animation import FuncAnimation

class FlowVisualizer:
    def __init__(self):
        """Initialize the flow visualizer."""
        self.fig = None
        self.axes = None
        self.lines = {}
    
    def _setup_plot(self):
        """Set up the plotting environment with subplots."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Supersonic Afterburner Flow Properties', fontsize=16)
        
        # Configure subplots
        self.axes[0, 0].set_title('Mach Number')
        self.axes[0, 0].set_xlabel('Position (m)')
        self.axes[0, 0].set_ylabel('Mach Number')
        self.axes[0, 0].grid(True)
        
        self.axes[0, 1].set_title('Pressure')
        self.axes[0, 1].set_xlabel('Position (m)')
        self.axes[0, 1].set_ylabel('Pressure (Pa)')
        self.axes[0, 1].grid(True)
        
        self.axes[1, 0].set_title('Temperature')
        self.axes[1, 0].set_xlabel('Position (m)')
        self.axes[1, 0].set_ylabel('Temperature (K)')
        self.axes[1, 0].grid(True)
        
        self.axes[1, 1].set_title('Density')
        self.axes[1, 1].set_xlabel('Position (m)')
        self.axes[1, 1].set_ylabel('Density (kg/m³)')
        self.axes[1, 1].grid(True)
        
        plt.tight_layout()
    
    def plot_results(self, results: Dict[str, np.ndarray], animate: bool = False):
        """Plot the flow properties from the simulation results."""
        if animate:
            self._animate_results(results)
        else:
            self._plot_static_results(results)
    
    def _plot_static_results(self, results: Dict[str, np.ndarray]):
        """Create static plots of the flow properties."""
        self._setup_plot()
        
        # Plot Mach number
        self.axes[0, 0].plot(results['x'], results['mach'], 'b-', label='Mach')
        self._mark_shocks(results, 0, 0)
        
        # Plot pressure
        self.axes[0, 1].plot(results['x'], results['pressure'], 'r-', label='Pressure')
        self._mark_shocks(results, 0, 1)
        
        # Plot temperature
        self.axes[1, 0].plot(results['x'], results['temperature'], 'g-', label='Temperature')
        self._mark_shocks(results, 1, 0)
        
        # Plot density
        self.axes[1, 1].plot(results['x'], results['density'], 'm-', label='Density')
        self._mark_shocks(results, 1, 1)
        
        plt.show()
    
    def _mark_shocks(self, results: Dict[str, np.ndarray], row: int, col: int):
        """Mark shock locations on the plots."""
        for x_shock in results['shock_locations']:
            self.axes[row, col].axvline(x=x_shock, color='k', linestyle='--', alpha=0.5)
    
    def _animate_results(self, results: Dict[str, np.ndarray]):
        """Create an animated visualization of the flow properties."""
        self._setup_plot()
        
        # Initialize lines for animation
        self.lines['mach'] = self.axes[0, 0].plot([], [], 'b-')[0]
        self.lines['pressure'] = self.axes[0, 1].plot([], [], 'r-')[0]
        self.lines['temperature'] = self.axes[1, 0].plot([], [], 'g-')[0]
        self.lines['density'] = self.axes[1, 1].plot([], [], 'm-')[0]
        
        # Set axis limits
        for ax in self.axes.flat:
            ax.set_xlim(0, results['x'][-1])
        
        self.axes[0, 0].set_ylim(0, 1.2 * np.max(results['mach']))
        self.axes[0, 1].set_ylim(0, 1.2 * np.max(results['pressure']))
        self.axes[1, 0].set_ylim(0, 1.2 * np.max(results['temperature']))
        self.axes[1, 1].set_ylim(0, 1.2 * np.max(results['density']))
        
        # Create animation
        anim = FuncAnimation(
            self.fig,
            self._update_animation,
            frames=len(results['x']),
            fargs=(results,),
            interval=50,
            blit=True
        )
        
        plt.show()
    
    def _update_animation(self, frame: int, results: Dict[str, np.ndarray]):
        """Update the animation frame."""
        x = results['x'][:frame]
        
        self.lines['mach'].set_data(x, results['mach'][:frame])
        self.lines['pressure'].set_data(x, results['pressure'][:frame])
        self.lines['temperature'].set_data(x, results['temperature'][:frame])
        self.lines['density'].set_data(x, results['density'][:frame])
        
        return list(self.lines.values())
