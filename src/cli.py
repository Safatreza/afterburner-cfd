import argparse
import yaml
from cerberus import Validator
import sys
from mesh2d import Mesh2D
from boundary_conditions import BoundaryConditions
from heat_profile_plugin import GaussianHeatProfile
from post_processing import PostProcessor
from compressible_ns_solver import CompressibleNSSolver

# Example schema (expand as needed)
schema = {
    'nx': {'type': 'integer', 'min': 2, 'required': True},
    'ny': {'type': 'integer', 'min': 2, 'required': True},
    'lx': {'type': 'float', 'min': 0, 'required': True},
    'ly': {'type': 'float', 'min': 0, 'required': True},
    'gamma': {'type': 'float', 'min': 1.0, 'required': True},
    'boundary_conditions': {'type': 'dict', 'required': True},
    'heat_source': {'type': 'dict', 'required': False},
    'turbulence': {'type': 'boolean', 'required': False},
    'combustion': {'type': 'boolean', 'required': False},
    'combustion_params': {'type': 'dict', 'required': False},
    'limiter': {'type': 'string', 'required': False},
}

def main():
    parser = argparse.ArgumentParser(description='Afterburner CFD Solver')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output', type=str, default='results.npz', help='Output file for results')
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    v = Validator(schema)
    if not v.validate(config):
        print('Config validation error:', v.errors)
        sys.exit(1)

    # Setup mesh and plugins
    mesh = Mesh2D(config['nx'], config['ny'], config['lx'], config['ly'])
    bc = BoundaryConditions(config, mesh)
    heat_plugin = GaussianHeatProfile(config, mesh)

    # Run solver
    solver = CompressibleNSSolver(config, mesh=mesh, bc=bc, heat_plugin=heat_plugin)
    solver.run(n_steps=config.get('n_steps', 100))
    post = PostProcessor(mesh, solver.fields)
    results = post.get_results()
    # Save results
    import numpy as np
    np.savez(args.output, **results)
    print(f'Results saved to {args.output}')

if __name__ == '__main__':
    main() 