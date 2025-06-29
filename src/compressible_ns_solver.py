import numpy as np
import logging

class CompressibleNSSolver:
    def __init__(self, config: dict, mesh=None, bc=None, heat_plugin=None):
        """Initialize the 2D compressible Navier-Stokes solver with config, mesh, BC, and heat plugin."""
        self.config = config
        self.mesh = mesh
        self.bc = bc
        self.heat_plugin = heat_plugin
        self.fields = {}
        self.turbulence = config.get('turbulence', False)
        self.combustion = config.get('combustion', False)
        self.heat_source = config.get('heat_source', None)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing solver')
        if self.mesh is None:
            from mesh2d import Mesh2D
            self.mesh = Mesh2D(config['nx'], config['ny'], config['lx'], config['ly'])
        self.setup_mesh()
        self.initialize_fields()

    def setup_mesh(self):
        # Use mesh object if provided
        if self.mesh is not None:
            self.nx = self.mesh.nx
            self.ny = self.mesh.ny
            self.lx = self.mesh.lx
            self.ly = self.mesh.ly
            self.dx = self.mesh.dx
            self.dy = self.mesh.dy
            self.x = self.mesh.x
            self.y = self.mesh.y
            self.X = self.mesh.X
            self.Y = self.mesh.Y
            self.mesh_tuple = (self.X, self.Y)
        else:
            # fallback
            nx = self.config.get('nx', 100)
            ny = self.config.get('ny', 50)
            lx = self.config.get('lx', 1.0)
            ly = self.config.get('ly', 0.5)
            self.nx, self.ny = nx, ny
            self.lx, self.ly = lx, ly
            self.dx = lx / (nx - 1)
            self.dy = ly / (ny - 1)
            self.x = np.linspace(0, lx, nx)
            self.y = np.linspace(0, ly, ny)
            self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
            self.mesh_tuple = (self.X, self.Y)

    def initialize_fields(self):
        nx, ny = self.nx, self.ny
        # Primitive variables: density, u, v, pressure, temperature
        rho0 = 1.0
        u0 = 500.0
        v0 = 0.0
        p0 = 101325.0
        T0 = 300.0
        gamma = self.config.get('gamma', 1.4)
        R = 287.0
        e0 = p0 / ((gamma - 1) * rho0)
        E0 = e0 + 0.5 * (u0**2 + v0**2)
        # Conservative variables
        self.fields['rho'] = np.ones((nx, ny)) * rho0
        self.fields['rhou'] = np.ones((nx, ny)) * (rho0 * u0)
        self.fields['rhov'] = np.ones((nx, ny)) * (rho0 * v0)
        self.fields['rhoE'] = np.ones((nx, ny)) * (rho0 * E0)
        # Primitive variables (for BCs and output)
        self.fields['u'] = np.ones((nx, ny)) * u0
        self.fields['v'] = np.ones((nx, ny)) * v0
        self.fields['p'] = np.ones((nx, ny)) * p0
        self.fields['T'] = np.ones((nx, ny)) * T0
        # Turbulence variables
        if self.turbulence:
            self.fields['k'] = np.ones((nx, ny)) * 1.0
            self.fields['omega'] = np.ones((nx, ny)) * 1.0

    def apply_boundary_conditions(self):
        if self.bc is not None:
            self.bc.apply(self.fields)
        else:
            # fallback to internal method
            bc = self.config.get('boundary_conditions', {})
            nx, ny = self.nx, self.ny
            gamma = self.config.get('gamma', 1.4)
            # Inlet (i=0)
            if bc.get('inlet', 'supersonic') == 'supersonic':
                rho0 = 1.0
                u0 = 500.0
                v0 = 0.0
                p0 = 101325.0
                T0 = 300.0
                e0 = p0 / ((gamma - 1) * rho0)
                E0 = e0 + 0.5 * (u0**2 + v0**2)
                self.fields['rho'][0, :] = rho0
                self.fields['rhou'][0, :] = rho0 * u0
                self.fields['rhov'][0, :] = rho0 * v0
                self.fields['rhoE'][0, :] = rho0 * E0
            # Outlet (i=-1)
            if bc.get('outlet', 'supersonic') == 'supersonic':
                for var in ['rho', 'rhou', 'rhov', 'rhoE']:
                    self.fields[var][-1, :] = self.fields[var][-2, :]
            # Walls (j=0, j=-1)
            if bc.get('walls', 'adiabatic') == 'adiabatic':
                # No-slip, adiabatic wall
                self.fields['rhou'][:, 0] = 0.0
                self.fields['rhou'][:, -1] = 0.0
                self.fields['rhov'][:, 0] = 0.0
                self.fields['rhov'][:, -1] = 0.0
                # Adiabatic: dT/dn = 0 (copy from adjacent cell)
                self.fields['rhoE'][:, 0] = self.fields['rhoE'][:, 1]
                self.fields['rhoE'][:, -1] = self.fields['rhoE'][:, -2]

    def minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Minmod slope limiter."""
        return np.where(np.abs(a) < np.abs(b), np.where(a * b > 0, a, 0), np.where(a * b > 0, b, 0))

    def van_leer(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Van Leer slope limiter."""
        return (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b)) / (1 + np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-12))

    def superbee(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Superbee slope limiter."""
        s = np.sign(a)
        return s * np.maximum(0, np.maximum(np.minimum(np.abs(a), 2*np.abs(b)), np.minimum(2*np.abs(a), np.abs(b))))

    def get_limiter(self) -> callable:
        """Return the slope limiter function based on config."""
        limiter_name = self.config.get('limiter', 'minmod').lower()
        if limiter_name == 'vanleer' or limiter_name == 'van_leer':
            return self.van_leer
        elif limiter_name == 'superbee':
            return self.superbee
        else:
            return self.minmod

    def compute_fluxes(self):
        # MUSCL reconstruction with selectable slope limiter
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        gamma = self.config.get('gamma', 1.4)
        limiter = self.get_limiter()
        # Get fields
        rho = self.fields['rho']
        rhou = self.fields['rhou']
        rhov = self.fields['rhov']
        rhoE = self.fields['rhoE']
        u = rhou / rho
        v = rhov / rho
        E = rhoE / rho
        p = (gamma - 1) * (rhoE - 0.5 * rho * (u**2 + v**2))
        # Prepare arrays
        flux_rho = np.zeros((nx, ny))
        flux_rhou = np.zeros((nx, ny))
        flux_rhov = np.zeros((nx, ny))
        flux_rhoE = np.zeros((nx, ny))
        # MUSCL slopes (x-direction)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                UL = np.array([rho[i-1,j], rhou[i-1,j], rhov[i-1,j], rhoE[i-1,j]])
                UC = np.array([rho[i,j], rhou[i,j], rhov[i,j], rhoE[i,j]])
                UR = np.array([rho[i+1,j], rhou[i+1,j], rhov[i+1,j], rhoE[i+1,j]])
                dU_L = UC - UL
                dU_R = UR - UC
                slope = limiter(dU_L, dU_R)
                U_L = UC + 0.5 * slope
                U_R = UR - 0.5 * slope
                uL = U_L[1] / U_L[0]; uR = U_R[1] / U_R[0]
                vL = U_L[2] / U_L[0]; vR = U_R[2] / U_R[0]
                EL = U_L[3] / U_L[0]; ER = U_R[3] / U_R[0]
                pL = (gamma - 1) * (U_L[3] - 0.5 * U_L[0] * (uL**2 + vL**2))
                pR = (gamma - 1) * (U_R[3] - 0.5 * U_R[0] * (uR**2 + vR**2))
                FL = np.array([U_L[1], U_L[1]*uL + pL, U_L[1]*vL, (U_L[3]+pL)*uL])
                FR = np.array([U_R[1], U_R[1]*uR + pR, U_R[1]*vR, (U_R[3]+pR)*uR])
                aL = np.sqrt(gamma * pL / U_L[0])
                aR = np.sqrt(gamma * pR / U_R[0])
                smax = np.max([np.abs(uL)+aL, np.abs(uR)+aR])
                F = 0.5*(FL + FR) - 0.5*smax*(U_R - U_L)
                flux_rho[i,j] -= (F[0] - flux_rho[i-1,j]) / dx
                flux_rhou[i,j] -= (F[1] - flux_rhou[i-1,j]) / dx
                flux_rhov[i,j] -= (F[2] - flux_rhov[i-1,j]) / dx
                flux_rhoE[i,j] -= (F[3] - flux_rhoE[i-1,j]) / dx
        # MUSCL slopes (y-direction)
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                UL = np.array([rho[i,j-1], rhou[i,j-1], rhov[i,j-1], rhoE[i,j-1]])
                UC = np.array([rho[i,j], rhou[i,j], rhov[i,j], rhoE[i,j]])
                UR = np.array([rho[i,j+1], rhou[i,j+1], rhov[i,j+1], rhoE[i,j+1]])
                dU_L = UC - UL
                dU_R = UR - UC
                slope = limiter(dU_L, dU_R)
                U_L = UC + 0.5 * slope
                U_R = UR - 0.5 * slope
                uL = U_L[1] / U_L[0]; uR = U_R[1] / U_R[0]
                vL = U_L[2] / U_L[0]; vR = U_R[2] / U_R[0]
                EL = U_L[3] / U_L[0]; ER = U_R[3] / U_R[0]
                pL = (gamma - 1) * (U_L[3] - 0.5 * U_L[0] * (uL**2 + vL**2))
                pR = (gamma - 1) * (U_R[3] - 0.5 * U_R[0] * (uR**2 + vR**2))
                GL = np.array([U_L[2], U_L[1]*vL, U_L[2]*vL + pL, (U_L[3]+pL)*vL])
                GR = np.array([U_R[2], U_R[1]*vR, U_R[2]*vR + pR, (U_R[3]+pR)*vR])
                aL = np.sqrt(gamma * pL / U_L[0])
                aR = np.sqrt(gamma * pR / U_R[0])
                smax = np.max([np.abs(vL)+aL, np.abs(vR)+aR])
                G = 0.5*(GL + GR) - 0.5*smax*(U_R - U_L)
                flux_rho[i,j] -= (G[0] - flux_rho[i,j-1]) / dy
                flux_rhou[i,j] -= (G[1] - flux_rhou[i,j-1]) / dy
                flux_rhov[i,j] -= (G[2] - flux_rhov[i,j-1]) / dy
                flux_rhoE[i,j] -= (G[3] - flux_rhoE[i,j-1]) / dy
        # Store fluxes
        self.fields['flux_rho'] = flux_rho
        self.fields['flux_rhou'] = flux_rhou
        self.fields['flux_rhov'] = flux_rhov
        self.fields['flux_rhoE'] = flux_rhoE

    def add_heat_source(self):
        nx, ny = self.nx, self.ny
        if self.heat_plugin is not None:
            Q = self.heat_plugin.get_heat_source(self.fields)
        else:
            X, Y = self.mesh_tuple
            Q = np.zeros((nx, ny))
            if self.heat_source and self.heat_source.get('type', 'gaussian') == 'gaussian':
                cx, cy = self.heat_source.get('center', (self.lx/2, self.ly/2))
                amp = self.heat_source.get('amplitude', 1e6)
                sx = self.heat_source.get('sigma_x', 0.05)
                sy = self.heat_source.get('sigma_y', 0.05)
                Q += amp * np.exp(-((X-cx)**2/(2*sx**2) + (Y-cy)**2/(2*sy**2)))
        if self.combustion:
            # Arrhenius single-step reaction model
            params = self.config.get('combustion_params', {})
            A = params.get('A', 1e6)
            Ea = params.get('Ea', 1e5)
            Y_fuel = params.get('Y_fuel', 0.05)
            T_ref = params.get('T_ref', 1500.0)
            R_univ = 8.314
            T = self.fields['T']
            rate = A * Y_fuel * np.exp(-Ea / (R_univ * np.maximum(T, 1.0)))
            Q += rate * (T > T_ref)
        self.fields['Q'] = Q

    def solve_turbulence(self):
        if not self.turbulence:
            return
        # Placeholder: simple explicit update for k-omega SST (not full model)
        k = self.fields['k']
        omega = self.fields['omega']
        # Example: decay (for demonstration)
        k *= 0.99
        omega *= 0.99
        self.fields['k'] = k
        self.fields['omega'] = omega

    def step(self):
        # Advance conservative variables by explicit Euler
        dt = 1e-6  # Example time step
        for var, flux in zip(['rho', 'rhou', 'rhov', 'rhoE'], ['flux_rho', 'flux_rhou', 'flux_rhov', 'flux_rhoE']):
            self.fields[var] += dt * self.fields.get(flux, 0)
        # Add heat source to energy
        if 'Q' in self.fields:
            self.fields['rhoE'] += dt * self.fields['Q']
        # Turbulence variables
        if self.turbulence:
            self.fields['k'] += 0
            self.fields['omega'] += 0

    def update_primitive_from_conservative(self):
        # Update u, v, p, T from conservative variables
        gamma = self.config.get('gamma', 1.4)
        R = 287.0
        rho = self.fields['rho']
        rhou = self.fields['rhou']
        rhov = self.fields['rhov']
        rhoE = self.fields['rhoE']
        u = rhou / rho
        v = rhov / rho
        E = rhoE / rho
        ke = 0.5 * (u**2 + v**2)
        e = E - ke
        p = (gamma - 1) * rho * e
        T = p / (rho * R)
        self.fields['u'] = u
        self.fields['v'] = v
        self.fields['p'] = p
        self.fields['T'] = T

    def run(self, n_steps: int = 1000) -> None:
        """Run the solver for n_steps time steps."""
        for step in range(n_steps):
            try:
                self.apply_boundary_conditions()
                self.compute_fluxes()
                if self.heat_source or self.combustion:
                    self.add_heat_source()
                if self.turbulence:
                    self.solve_turbulence()
                self.step()
                self.update_primitive_from_conservative()
                if step % 10 == 0:
                    self.logger.info(f'Step {step} completed')
            except Exception as e:
                self.logger.error(f'Error at step {step}: {e}')
                raise

    def get_results(self):
        # Return a dictionary of key fields for visualization/post-processing
        results = {
            'x': self.x,
            'y': self.y,
            'rho': self.fields['rho'],
            'rhou': self.fields['rhou'],
            'rhov': self.fields['rhov'],
            'rhoE': self.fields['rhoE'],
            'u': self.fields['u'],
            'v': self.fields['v'],
            'p': self.fields['p'],
            'T': self.fields['T']
        }
        if self.turbulence:
            results['k'] = self.fields['k']
            results['omega'] = self.fields['omega']
        return results 