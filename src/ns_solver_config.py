config = {
    'nx': 100,
    'ny': 50,
    'lx': 1.0,
    'ly': 0.5,
    'gamma': 1.4,
    'prandtl': 0.72,
    'reynolds': 1e5,
    'turbulence': True,
    'combustion': False,
    'heat_source': {
        'type': 'gaussian',
        'center': (0.5, 0.25),
        'amplitude': 1e6,
        'sigma_x': 0.05,
        'sigma_y': 0.05
    },
    'boundary_conditions': {
        'inlet': 'supersonic',
        'outlet': 'supersonic',
        'walls': 'adiabatic'
    },
    'limiter': 'minmod',  # options: minmod, vanleer, superbee
    'combustion_params': {
        'A': 1e6,         # pre-exponential factor
        'Ea': 1e5,        # activation energy (J/mol)
        'Y_fuel': 0.05,   # fuel mass fraction
        'T_ref': 1500.0   # reference temp (K)
    }
} 