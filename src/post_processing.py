import numpy as np

class PostProcessor:
    def __init__(self, mesh, fields):
        self.mesh = mesh
        self.fields = fields
    def get_results(self):
        results = {
            'x': self.mesh.x,
            'y': self.mesh.y,
            'rho': self.fields['rho'],
            'rhou': self.fields['rhou'],
            'rhov': self.fields['rhov'],
            'rhoE': self.fields['rhoE'],
            'u': self.fields['u'],
            'v': self.fields['v'],
            'p': self.fields['p'],
            'T': self.fields['T']
        }
        if 'k' in self.fields:
            results['k'] = self.fields['k']
        if 'omega' in self.fields:
            results['omega'] = self.fields['omega']
        return results 