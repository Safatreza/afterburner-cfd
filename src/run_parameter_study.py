from parameter_study import ParameterStudy
import numpy as np
from pathlib import Path

def main():
    # Define the base case directory
    base_case_dir = Path("cases/base_case")
    
    # Create parameter study instance
    study = ParameterStudy(
        base_case_dir=str(base_case_dir),
        study_name="injector_angle_study"
    )
    
    # Define parameter sets to study
    injector_angles = np.linspace(30, 60, 7)  # 7 angles from 30° to 60°
    v_gutter_shapes = [
        {"type": "sharp", "angle": 45},
        {"type": "rounded", "radius": 0.02},
        {"type": "blunt", "width": 0.05}
    ]
    
    # Create parameter sets
    parameter_sets = []
    for angle in injector_angles:
        for shape in v_gutter_shapes:
            params = {
                # Mesh parameters
                "injectorAngle": angle,
                "vGutterType": shape["type"],
                "refinementLevel": 3,
                
                # Geometry parameters
                "injectorDiameter": 0.01,
                "vGutterWidth": 0.05,
                "domainLength": 1.0,
                
                # Simulation parameters
                "inletVelocity": 100.0,
                "inletTemperature": 300.0,
                "turbulenceIntensity": 0.05,
                
                # Solver parameters
                "maxCo": 0.5,
                "endTime": 1.0,
                "writeInterval": 0.1
            }
            
            # Add shape-specific parameters
            if shape["type"] == "sharp":
                params["vGutterAngle"] = shape["angle"]
            elif shape["type"] == "rounded":
                params["vGutterRadius"] = shape["radius"]
            elif shape["type"] == "blunt":
                params["vGutterWidth"] = shape["width"]
            
            parameter_sets.append(params)
    
    # Define custom mesh and simulation commands if needed
    mesh_commands = [
        "blockMesh",
        "snappyHexMesh -overwrite",
        "checkMesh"
    ]
    
    simulation_commands = [
        "simpleFoam",
        "postProcess -func 'mag(U)'",
        "postProcess -func 'mag(grad(U))'"
    ]
    
    # Run the parameter study
    study.run_parameter_study(
        parameter_sets=parameter_sets,
        mesh_commands=mesh_commands,
        simulation_commands=simulation_commands
    )
    
    # Generate summary report
    study.generate_summary_report()

if __name__ == "__main__":
    main() 