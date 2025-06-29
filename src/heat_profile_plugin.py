import numpy as np

class HeatProfilePlugin:
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
    def get_heat_source(self, fields):
        raise NotImplementedError

class GaussianHeatProfile(HeatProfilePlugin):
    def get_heat_source(self, fields):
        X, Y = self.mesh.X, self.mesh.Y
        params = self.config.get('heat_source', {})
        cx, cy = params.get('center', (self.mesh.lx/2, self.mesh.ly/2))
        amp = params.get('amplitude', 1e6)
        sx = params.get('sigma_x', 0.05)
        sy = params.get('sigma_y', 0.05)
        return amp * np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2))) 