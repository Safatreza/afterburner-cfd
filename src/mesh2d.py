import numpy as np

class Mesh2D:
    def __init__(self, nx: int, ny: int, lx: float, ly: float):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij') 