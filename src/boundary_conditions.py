import numpy as np

class BoundaryConditions:
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh

    def apply(self, fields):
        nx, ny = self.mesh.nx, self.mesh.ny
        gamma = self.config.get('gamma', 1.4)
        bc = self.config.get('boundary_conditions', {})
        # Inlet (i=0)
        if bc.get('inlet', 'supersonic') == 'supersonic':
            rho0 = 1.0
            u0 = 500.0
            v0 = 0.0
            p0 = 101325.0
            e0 = p0 / ((gamma - 1) * rho0)
            E0 = e0 + 0.5 * (u0**2 + v0**2)
            fields['rho'][0, :] = rho0
            fields['rhou'][0, :] = rho0 * u0
            fields['rhov'][0, :] = rho0 * v0
            fields['rhoE'][0, :] = rho0 * E0
        # Outlet (i=-1)
        if bc.get('outlet', 'supersonic') == 'supersonic':
            for var in ['rho', 'rhou', 'rhov', 'rhoE']:
                fields[var][-1, :] = fields[var][-2, :]
        # Walls (j=0, j=-1)
        if bc.get('walls', 'adiabatic') == 'adiabatic':
            fields['rhou'][:, 0] = 0.0
            fields['rhou'][:, -1] = 0.0
            fields['rhov'][:, 0] = 0.0
            fields['rhov'][:, -1] = 0.0
            fields['rhoE'][:, 0] = fields['rhoE'][:, 1]
            fields['rhoE'][:, -1] = fields['rhoE'][:, -2] 