import numpy as np
from pathlib import Path
import json
from src.validation.experimental_comparison import ExperimentalComparison
from src.validation.textbook_cases import LaminarPoiseuilleFlow, BlasiusBoundaryLayer, TextbookCaseManager

def create_test_data():
    """Create test data for validation."""
    # Create test experimental data
    x = np.linspace(0, 1, 100)
    
    # Generate "experimental" data with some noise
    u_exp = 1 - x**2 + np.random.normal(0, 0.05, len(x))
    p_exp = -x + np.random.normal(0, 0.02, len(x))
    
    # Generate "CFD" data with some systematic error
    u_cfd = 1 - x**2 + 0.1 * np.sin(2 * np.pi * x)
    p_cfd = -x + 0.05 * np.cos(2 * np.pi * x)
    
    # Save experimental data
    exp_data = {
        'data': {
            'u': u_exp.tolist(),
            'p': p_exp.tolist()
        },
        'uncertainty': {
            'u': [0.05] * len(x),
            'p': [0.02] * len(x)
        }
    }
    
    with open('test_data/experimental_data.json', 'w') as f:
        json.dump(exp_data, f, indent=4)
    
    return {
        'x': x,
        'cfd_data': {
            'u': u_cfd,
            'p': p_cfd
        }
    }

def test_experimental_validation():
    """Test experimental data validation."""
    print("\nTesting Experimental Validation...")
    
    # Create test data
    test_data = create_test_data()
    
    # Initialize experimental comparison
    comparison = ExperimentalComparison('test_data/experimental_data.json')
    
    # Compare CFD results with experimental data
    metrics = comparison.compare_with_cfd(
        test_data['cfd_data'],
        save_dir='test_results/experimental'
    )
    
    # Generate validation report
    report = comparison.generate_validation_report(
        test_data['cfd_data'],
        save_dir='test_results/experimental'
    )
    
    print("\nExperimental Validation Report:")
    print(report)

def test_textbook_validation():
    """Test textbook case validation."""
    print("\nTesting Textbook Case Validation...")
    
    # Create test data
    x = np.linspace(-1, 1, 100)
    
    # Create Poiseuille flow case
    poiseuille = LaminarPoiseuilleFlow(
        dp_dx=-1.0,  # Pressure gradient
        mu=1.0,      # Dynamic viscosity
        rho=1.0,     # Density
        h=1.0        # Channel height
    )
    
    # Create Blasius boundary layer case
    blasius = BlasiusBoundaryLayer(
        u_inf=1.0,   # Free stream velocity
        nu=1e-5      # Kinematic viscosity
    )
    
    # Create manager and add cases
    manager = TextbookCaseManager()
    manager.add_case(poiseuille)
    manager.add_case(blasius)
    
    # Generate CFD-like data with some error
    poiseuille_cfd = {
        'u': poiseuille.get_reference_solution(x)['u'] + 0.1 * np.sin(2 * np.pi * x),
        'p': poiseuille.get_reference_solution(x)['p'] + 0.05 * np.cos(2 * np.pi * x)
    }
    
    blasius_cfd = {
        'u': blasius.get_reference_solution(x)['u'] + 0.1 * np.exp(-x),
        'v': blasius.get_reference_solution(x)['v'] + 0.05 * np.exp(-x)
    }
    
    # Validate results
    results = manager.validate_all(
        cfd_data={
            'LaminarPoiseuilleFlow': poiseuille_cfd,
            'BlasiusBoundaryLayer': blasius_cfd
        },
        x_coords={
            'LaminarPoiseuilleFlow': x,
            'BlasiusBoundaryLayer': x
        },
        save_dir='test_results/textbook'
    )
    
    # Generate summary report
    report = manager.generate_summary_report(results, 'test_results/textbook')
    
    print("\nTextbook Case Validation Report:")
    print(report)

def main():
    """Run all validation tests."""
    # Create test directories
    Path('test_data').mkdir(exist_ok=True)
    Path('test_results/experimental').mkdir(parents=True, exist_ok=True)
    Path('test_results/textbook').mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_experimental_validation()
    test_textbook_validation()

if __name__ == '__main__':
    main() 