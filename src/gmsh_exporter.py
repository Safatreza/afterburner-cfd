import numpy as np

def get_mesh_parameters(height=1.0, ny=20):
    """
    Return mesh parameters for Gmsh export.
    Args:
        height: height of the 2D domain (default 1.0)
        ny: number of points in y direction (default 20)
    Returns:
        tuple: (height, ny)
    """
    return height, ny

def write_gmsh_2d(results, output_path, height=1.0, ny=20):
    """
    Export 1D CFD results as a 2D mesh (extruded in y) with field data in Gmsh .msh format.
    Args:
        results: dict with keys 'x', 'mach', 'pressure', 'temperature', 'density', 'velocity'
        output_path: path to save the .msh file
        height: height of the 2D domain (default 1.0)
        ny: number of points in y direction (default 20)
    """
    height, ny = get_mesh_parameters(height, ny)
    x = results['x']
    nx = len(x)
    y = np.linspace(0, height, ny)
    nodes = []
    node_id = 1
    node_map = {}
    for j in range(ny):
        for i in range(nx):
            nodes.append((node_id, x[i], y[j], 0.0))
            node_map[(i, j)] = node_id
            node_id += 1
    elements = []
    elem_id = 1
    for j in range(ny-1):
        for i in range(nx-1):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append((elem_id, n1, n2, n3, n4))
            elem_id += 1
    # Write Gmsh v2 ASCII format
    with open(output_path, 'w') as f:
        f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        f.write(f'$Nodes\n{len(nodes)}\n')
        for n in nodes:
            f.write(f'{n[0]} {n[1]} {n[2]} {n[3]}\n')
        f.write('$EndNodes\n')
        f.write(f'$Elements\n{len(elements)}\n')
        for e in elements:
            # 3 = 4-node quadrangle
            f.write(f'{e[0]} 3 2 0 1 {e[1]} {e[2]} {e[3]} {e[4]}\n')
        f.write('$EndElements\n')
        # Write field data as node data
        for field in ['mach', 'pressure', 'temperature', 'density', 'velocity']:
            f.write(f'$NodeData\n1\n"{field}"\n1\n0.0\n3\n0\n1\n{len(nodes)}\n')
            arr = results[field]
            for j in range(ny):
                for i in range(nx):
                    val = arr[i]  # 1D field extruded in y
                    node_idx = node_map[(i, j)]
                    f.write(f'{node_idx} {val}\n')
            f.write('$EndNodeData\n')
    print(f"Gmsh .msh file written to {output_path}") 