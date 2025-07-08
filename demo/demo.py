# demo/demo.py
# Minimal, dependency-free demo of afterburner Mach number profile

"""
Demo: Supersonic Afterburner CFD Simulation (Minimal Version)

Purpose:
    This demo provides a minimal, dependency-light example of how a Mach number profile evolves along a supersonic afterburner. It is intended for educational purposes and to illustrate the core concept of the main project without requiring any external CFD libraries or complex setup.

Blocks of Code:
    1. simple_mach_profile: Generates a linear Mach number profile along the afterburner length.
    2. plot_profile: Visualizes the Mach number profile using matplotlib.
    3. main: Runs the demo, prints the profile, and shows the plot.

Project Purpose:
    The full project simulates compressible flow in a supersonic afterburner, including heat addition, shock detection, validation against experimental/textbook data, and visualization. This demo is a simplified, dependency-free illustration of the Mach number evolution concept.
"""

def simple_mach_profile(length=1.0, points=10, mach_inlet=1.2, mach_exit=2.0):
    """Generate a linear Mach number profile along the afterburner."""
    x = [i * length / (points - 1) for i in range(points)]
    mach = [mach_inlet + (mach_exit - mach_inlet) * (xi / length) for xi in x]
    return x, mach

def plot_profile(x, mach):
    """Plot the Mach number profile using matplotlib."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(x, mach, marker='o', color='b')
    plt.title('Mach Number Profile Along Afterburner (Demo)')
    plt.xlabel('Position (m)')
    plt.ylabel('Mach Number')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    print("Afterburner Mach Number Profile (Demo)")
    x, mach = simple_mach_profile()
    print(f"{'Position (m)':>15} | {'Mach Number':>12}")
    print("-" * 30)
    for xi, mi in zip(x, mach):
        print(f"{xi:15.3f} | {mi:12.3f}")
    plot_profile(x, mach)

if __name__ == "__main__":
    main() 