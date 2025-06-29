from setuptools import setup, find_packages

setup(
    name='afterburner_cfd',
    version='0.1.0',
    description='2D Compressible Navier-Stokes CFD Solver with Modular Architecture',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy', 'matplotlib', 'pyyaml', 'cerberus', 'flask', 'streamlit', 'reportlab', 'plotly'
    ],
    entry_points={
        'console_scripts': [
            'afterburner-cfd=main:main',
        ],
    },
    include_package_data=True,
    python_requires='>=3.8',
) 