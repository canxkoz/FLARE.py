#
import torch
import numpy as np
import torch_geometric as pyg

import os
import shutil

__all__ = [
    # PyVista
    'visualize_pyv',
    'visualize_timeseries_pyv',
    'mesh_pyv',
    'write_pvd',
    # Matplotlib
    'visualize_mpl',
    'verify_connectivity',
    # Open3D
    'visualize_o3d',
    # TriMesh
    'visualize_tri',
]

#======================================================================#
# PyVista
#======================================================================#
def add_field(mesh, graph, mesh_field: str, graph_field: str, istep=None):
    if istep is None:
        mesh.point_data[mesh_field] = graph[graph_field].numpy(force=True)
    else:
        mesh.point_data[mesh_field] = graph[graph_field][istep].numpy(force=True)
    return

def try_add_field(mesh, graph, mesh_field: str, graph_field: str, istep=None):
    if istep is None:
        try:
            mesh.point_data[mesh_field] = graph[graph_field].numpy(force=True)
        except:
            pass
    else:
        try:
            mesh.point_data[mesh_field] = graph[graph_field][istep].numpy(force=True)
        except:
            pass
    return

def visualize_pyv(graph, out_file=None):
    mesh = mesh_pyv(graph.pos, graph.elems)

    add_field(mesh, graph, 'disp', 'disp')
    add_field(mesh, graph, 'temp', 'temp')
    add_field(mesh, graph, 'vmstr', 'vmstr')

    try_add_field(mesh, graph, 'sdf_x', 'sdf_x')

    try_add_field(mesh, graph, 'source_normalized', 'x')
    try_add_field(mesh, graph, 'target_normalized', 'y')
    try_add_field(mesh, graph, 'pred_normalized', 'yh')
    try_add_field(mesh, graph, 'error_normalized', 'e')
    try_add_field(mesh, graph, 'pred', 'yp')
    try_add_field(mesh, graph, 'mask', 'mask')

    if out_file is not None:
        mesh.save(out_file)

    return mesh

def visualize_timeseries_pyv(dataset, out_dir, icase=None, merge=None, name=None):
    N = len(dataset)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for i in range(N):
        graph = dataset[i]
        mesh = mesh_pyv(graph.pos, graph.elems)

        if graph.disp.ndim == 3: # merge
            istep = graph.metadata['time_step']
            add_field(mesh, graph, 'disp', 'disp', istep)
            add_field(mesh, graph, 'temp', 'temp', istep)
            add_field(mesh, graph, 'vmstr', 'vmstr', istep)
            try_add_field(mesh, graph, 'sdf_x', 'sdf_x', istep)
        else: # == 2
            add_field(mesh, graph, 'disp', 'disp')
            add_field(mesh, graph, 'temp', 'temp')
            add_field(mesh, graph, 'vmstr', 'vmstr')
            try_add_field(mesh, graph, 'sdf_x', 'sdf_x')

        try_add_field(mesh, graph, 'source_normalized', 'x')
        try_add_field(mesh, graph, 'target_normalized', 'y')
        try_add_field(mesh, graph, 'pred_normalized', 'yh')
        try_add_field(mesh, graph, 'error_normalized', 'e')
        try_add_field(mesh, graph, 'pred', 'yp')
        try_add_field(mesh, graph, 'mask', 'mask')

        try_add_field(mesh, graph, 'mask_bulk', 'mask_bulk')

        mesh.save(os.path.join(out_dir, f'data{str(i).zfill(2)}.vtu'))

    if name is None:
        name = 'merged' if merge else 'series'
    if icase is not None:
        name = name + str(icase).zfill(2)

    pvd_file = os.path.join(out_dir, f'{name}.pvd')
    write_pvd(pvd_file, N, 'data')

    return

def mesh_pyv(pos: torch.Tensor, elems: torch.Tensor):
    import pyvista as pv

    PPE = 8 # points per element
    celltype = pv.CellType.HEXAHEDRON
    celltypes = np.full(elems.shape[0], celltype, dtype=np.uint8)

    pos = pos.numpy(force=True)   # [Nv, 3]
    elm = elems.numpy(force=True) # [Ne, PPE]

    NV = pos.shape[0]
    NE = elm.shape[0]

    points = pos
    cells  = np.concat((np.full((NE,1), PPE), elm), axis=1)#.ravel()
    mesh   = pv.UnstructuredGrid(cells, celltypes, points)

    return mesh

def write_pvd(pvd_file, N, vtu_name):
    with open(pvd_file, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        for i in range(N):
            f.write(f'    <DataSet timestep="{i}" group="" part="0" file="{vtu_name}{str(i).zfill(2)}.vtu"/>\n')
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

    return

#======================================================================#
# Matplotlib
#======================================================================#
def visualize_mpl(graph: pyg.data.Data, val: str,
    make_edge=True, max_edges=100_000, cmap='jet',
):
    import matplotlib.pyplot as plt

    pos = graph.pos.numpy(force=True)
    val = getattr(graph, val).numpy(force=True)
    edges = graph.edge_index.sort(dim=0).values.unique(dim=1).numpy(force=True).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)

    # colors
    cmap = plt.get_cmap(cmap)
    lb, ub = min(val), max(val)
    if (ub - lb) < 1e-6:
        cvals = val * 0 + 1
    else:
        cvals = (val - lb) / (ub - lb)
    colors = cmap(cvals.reshape(-1)) # [Nv, 4]

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors, s=2)
    if make_edge:
        for (i, edge) in enumerate(edges):
            if i >= max_edges:
                print(f"Plotting {max_edges} / {edges.shape[0]} edges.")
                break
            start, end = edge
            ax.plot([pos[start][0], pos[end][0]],
                    [pos[start][1], pos[end][1]],
                    [pos[start][2], pos[end][2]],
                    c='gray', linewidth=0.5)

    return fig

def verify_connectivity(graph: pyg.data.Data, elem_idx=1):
    import matplotlib.pyplot as plt

    pos   = graph.pos.numpy(force=True)
    elems = graph.elems.numpy(force=True)
    edges = graph.edge_index.sort(dim=0).values.unique(dim=1).numpy(force=True).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.grid(False)

    # only plot one element
    v = elems[elem_idx]
    mask  = np.isin(edges, v).all(axis=1)
    edges = edges[mask]

    # remake edges for verification
    # connectivity = [                    # hexa8 elements
    #     (0, 1), (1, 2), (2, 3), (3, 0), # cube base
    #     (4, 5), (5, 6), (6, 7), (7, 4), # cube top
    #     (0, 4), (1, 5), (2, 6), (3, 7), # vertical edges
    # ]
    #
    # elem = graph.elems[elem_idx]
    # edges = set()
    # for (i, j) in connectivity:
    #     edge = (elem[i].item(), elem[j].item())
    #     edges.add(edge)

    ax.scatter(pos[v,0], pos[v,1], pos[v,2], c='red', s=20)
    for edge in edges:
        start, end = edge
        ax.plot([pos[start][0], pos[end][0]],
            [pos[start][1], pos[end][1]],
            [pos[start][2], pos[end][2]],
            c='black', linewidth=1.0)

    print(f'Element {elem_idx} has {len(edges)} edges.')
    return fig

#======================================================================#
def tri_faces(elems):
    '''
    break up 6 faces of an hexa8 element into 12 triangles
    '''
    tri_idx = np.array([[0,0,0,0,0,0,6,6,6,6,6,6],
                        [2,3,7,4,1,5,2,3,1,5,4,7],
                        [1,2,3,7,5,4,3,7,2,1,5,4]], dtype=int)

    tri_faces = np.zeros([12*elems.shape[0],3], dtype=int)
    for i, elem in enumerate(elems):
        tri_faces[i*12:(i+1)*12,:] = elem[tri_idx].T

    return tri_faces

def quad_faces(elems):
    '''
    break out the 6 faces of an hexa8 element
    '''
    quad_idx = np.array([[0,4,0,1,2,3,],
                         [1,5,1,2,3,0,],
                         [2,6,5,6,7,4,],
                         [3,7,4,5,6,7,]], dtype=int)

    quad_faces = np.zeros([6*elems.shape[0],4], dtype=int)
    for i, elem in enumerate(elems):
        quad_faces[i*12:(i+1)*12,:] = elem[quad_idx].T

    return quad_faces

#======================================================================#
# Open3D
#======================================================================#
def visualize_o3d(graph: pyg.data.Data, val: str, imagefile: str):
    import open3d as o3d

    pos = graph.pos.numpy(force=True)
    val = getattr(graph, val).numpy(force=True)
    elm = graph.elems.numpy(force=True)

    # MESH
    tris = tri_faces(elems)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(float))
    mesh.triangles = o3d.utility.Vector3iVector(tris.astype(np.int32))

    cmap = plt.get_cmap(cmap)
    lb, ub = min(values), max(values)
    values = (values - lb) / (ub - lb)
    colors = cmap(values.reshape(-1)) # [Nv, 4]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:,:3].astype(float))

    # VIS
    o3d.visualization.webrtc_server.enable_webrtc()

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    # vis.add_geometry(mesh)
    # # vis.poll_events()
    # # vis.update_renderer()
    # vis.capture_screen_image(imagefile)
    # vis.destroy_window()

    return

#======================================================================#
# TriMesh
#======================================================================#
def visualize_tri(graph: pyg.data.Data, val: str, imagefile: str, cmap='jet'):
    import trimesh
    import matplotlib.pyplot as plt

    elems = graph.elems.numpy(force=True)
    faces = tri_faces(elems)
    graph.face = tri_faces(elems)
    mesh  = pyg.utils.to_trimesh(graph)

    cmap = plt.get_cmap(cmap)
    lb, ub = min(val), max(val)
    cvals = (val - lb) / (ub - lb)
    colors = cmap(cvals.reshape(-1)) # [Nv, 4]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(
        mesh.vertices[:,0], mesh.vertices[:,1], mesh.vertices[:,2],
        triangles=mesh.faces,
        cmap='jet',
        edgecolor='black', linewidth=0.2,
    )

    return fig
#======================================================================#
#
