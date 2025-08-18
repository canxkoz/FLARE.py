#
import os
import numpy as np
import scipy

__all__ = [
    'sdf_features',
]

#======================================================================#
def make_faces(elems):
    """
    Get all faces (6 per hex, 4 nodes per face)
    """
    assert elems.ndim == 2 and elems.shape[1] == 8

    faces = np.hstack([
        elems[:, [0, 1, 2, 3]],  # bottom
        elems[:, [4, 5, 6, 7]],  # top
        elems[:, [0, 1, 5, 4]],  # front
        elems[:, [1, 2, 6, 5]],  # right
        elems[:, [2, 3, 7, 6]],  # back
        elems[:, [3, 0, 4, 7]]   # left
    ])

    faces = faces.reshape(-1, 4)
    return faces

def quads_to_triangles(faces):
    """
    Convert quads to triangles
    """
    assert faces.ndim == 2 and faces.shape[1] == 4

    triangles = np.vstack([
        faces[:, [0, 1, 2]],
        faces[:, [0, 2, 3]]
    ])

    return triangles

#======================================================================#
def select_tjt_large(faces, pos):
    """
    Select large faces at T-Junctions
    """

    # face centers and normals
    face_centers = np.mean(pos[faces], axis=1)
    v1 = pos[faces[:, 1]] - pos[faces[:, 0]]
    v2 = pos[faces[:, 2]] - pos[faces[:, 0]]
    face_normals = np.cross(v1, v2)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    
    # face edge sizes
    L = np.linalg.norm(pos[faces[:, 0]] - pos[faces[:, 1]], axis=1)
    W = np.linalg.norm(pos[faces[:, 1]] - pos[faces[:, 2]], axis=1)
    
    tree = scipy.spatial.KDTree(face_centers)

    # Create a mask for large faces at T-Junctions (initially all False)
    is_tjt_large = np.zeros(len(faces), dtype=bool)
    
    # For each face, find nearby faces and check if they're opposite
    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):

        # Find faces within a distance based on expected center distance in a T-junction
        # we are grabbing neighbors only for the large face at a T-junction
        tjt_center_f2f_dist = np.sqrt(L[i]**2 + W[i]**2) / 4

        # expected distance between adjacent faces
        tjt_center_adj_dist1 = np.sqrt(9*L[i]**2 + 1*W[i]**2) / 4
        tjt_center_adj_dist2 = np.sqrt(1*L[i]**2 + 9*W[i]**2) / 4
        tjt_center_adj_dist = np.max([tjt_center_adj_dist1, tjt_center_adj_dist2])

        neighbors = tree.query_ball_point(center, r=tjt_center_adj_dist * 1.001)
        # neighbors = tree.query_ball_point(center, r=tjt_center_f2f_dist * 1.001)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or is_tjt_large[j]:
                continue
                
            # Check if normals align
            # ideally normals should be opposite, but orientation isn't guaranteed
            if not np.abs(np.dot(normal, face_normals[j])) > 0.999:
                continue
            
            face_center_disp = face_centers[j] - center
            face_center_dist = np.linalg.norm(face_center_disp)

            # if differs from expected
            # tjt_center_f2f_dist or
            # tjt_center_adj_dist1 or
            # tjt_center_adj_dist2
            # then it is not a T-junction
            if (
                not (tjt_center_f2f_dist  * 0.999 < face_center_dist < tjt_center_f2f_dist  * 1.001) and
                not (tjt_center_adj_dist1 * 0.999 < face_center_dist < tjt_center_adj_dist1 * 1.001) and
                not (tjt_center_adj_dist2 * 0.999 < face_center_dist < tjt_center_adj_dist2 * 1.001)
            ):
                continue
            
            # get direction bw face centers
            # if direction is axis-aligned, then continue
            face_center_direction = face_center_disp / face_center_dist
            if (
                np.abs(face_center_direction[0]) > 0.999 or
                np.abs(face_center_direction[1]) > 0.999 or
                np.abs(face_center_direction[2]) > 0.999
            ):
                continue
            
            # if we are here, then face [i] is a large face at a T-junction
            is_tjt_large[i] = True
            
    return is_tjt_large

class PointHash:
    """
    A class for hashing 3D point coordinates to indices using integer coordinates
    """
    def __init__(self, pos, scale=None):
        """
        Initialize the PointHash with point coordinates
        Args:
            pos: (N, 3) array of point coordinates
            scale: scaling factor for converting to integer coordinates
        """
        if scale is None:
            scale = 1e6

        self.scale = scale
        self._hash = self._create_hash(pos)
        
    def _create_hash(self, pos):
        """
        Create the internal hash table
        Args:
            pos: (N, 3) array of point coordinates
        Returns:
            Dictionary mapping tuple of rounded coordinates to indices
        """
        int_pos = np.round(pos * self.scale).astype(np.int64)

        # Check for collisions
        if len(np.unique(int_pos, axis=0)) < len(pos):
            raise ValueError(f"PointHash: scale = {self.scale} is too small: Collisions detected in hashing. Min/max scaled value: {np.min(int_pos)}/{np.max(int_pos)}")
    
        return {tuple(coords): idx for idx, coords in enumerate(int_pos)}
    
    def check_point_existence(self, point):
        """
        Check if a point exists in the hash table
        Args:
            point: (3,) array of point coordinates
        Returns:
            index of point if exists, None otherwise
        """
        int_point = tuple(np.round(point * self.scale).astype(np.int64))
        return self._hash.get(int_point, None)
    
    def update_hash(self, point, index):
        """
        Update the hash with a new point and index
        Args:
            hash: PointHash object
            point: (3,) array of point coordinates
            index: integer index of the point
        """
        int_point = tuple(np.round(point * self.scale).astype(np.int64))
        self._hash[int_point] = index
        return

def break_tjunctions(faces, pos, debug=None):
    """
    Break faces at T-Junctions
    """
    
    L = np.linalg.norm(pos[faces[:, 0]] - pos[faces[:, 1]], axis=1)
    W = np.linalg.norm(pos[faces[:, 1]] - pos[faces[:, 2]], axis=1)
    scale = 1 / np.min(np.vstack([L, W]))
    hash = PointHash(pos, scale=scale * 2)

    # get large faces at T-Junctions
    is_tjt_large = select_tjt_large(faces, pos)

    # break faces at T-Junctions
    faces_new = faces[~is_tjt_large]
    pos_new = pos.copy()

    # break face at T-Junction
    for face in faces[is_tjt_large]:
        # get positions
        p0, p1, p2, p3 = pos[face]
        p4 = np.mean(pos[face[[0, 1]]], axis=0) # get edge centers
        p5 = np.mean(pos[face[[1, 2]]], axis=0)
        p6 = np.mean(pos[face[[2, 3]]], axis=0)
        p7 = np.mean(pos[face[[3, 0]]], axis=0)
        p8 = np.mean(pos[face], axis=0) # get face center

        # get indices
        i0, i1, i2, i3 = face
        # check if p4-p7 already exist in pos.
        i4 = hash.check_point_existence(p4)
        i5 = hash.check_point_existence(p5)
        i6 = hash.check_point_existence(p6)
        i7 = hash.check_point_existence(p7)
        i8 = hash.check_point_existence(p8)

        if i4 is None:
            pos_new = np.vstack([pos_new, p4])
            i4 = len(pos_new) - 1
            hash.update_hash(p4, i4)
        if i5 is None:
            pos_new = np.vstack([pos_new, p5])
            i5 = len(pos_new) - 1
            hash.update_hash(p5, i5)
        if i6 is None:
            pos_new = np.vstack([pos_new, p6])
            i6 = len(pos_new) - 1
            hash.update_hash(p6, i6)
        if i7 is None:
            pos_new = np.vstack([pos_new, p7])
            i7 = len(pos_new) - 1
            hash.update_hash(p7, i7)
        if i8 is None:
            pos_new = np.vstack([pos_new, p8])
            i8 = len(pos_new) - 1
            hash.update_hash(p8, i8)
    
        # get new faces
        faces_new = np.vstack([
            faces_new,
            np.array([i0, i4, i8, i7]),
            np.array([i4, i1, i5, i8]),
            np.array([i8, i5, i2, i6]),
            np.array([i7, i8, i6, i3]),
        ])

    assert len(faces_new) == len(faces[~is_tjt_large]) + 4 * np.sum(is_tjt_large)

    if debug is not None:
        print(f"break_tjunctions: added {len(faces_new) - len(faces)}/{len(faces)} faces, {len(pos_new) - len(pos)}/{len(pos)} nodes")
        
    # # ensure distance bw points is not too small
    # tree = scipy.spatial.KDTree(pos_new)
    # dists, idxs = tree.query(pos_new, k=2)
    # dists = dists[:, 1:]
    # idxs = idxs[:, 1:]
    # assert np.min(dists) > 1e-4
    
    return pos_new, faces_new

def omit_internal_tjunctions(faces, pos, debug=None):
    """
    Omit faces at T-Junctions
    """

    # face centers and normals
    face_centers = np.mean(pos[faces], axis=1)
    v1 = pos[faces[:, 1]] - pos[faces[:, 0]]
    v2 = pos[faces[:, 2]] - pos[faces[:, 0]]
    face_normals = np.cross(v1, v2)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    
    # face edge sizes
    L = np.linalg.norm(pos[faces[:, 0]] - pos[faces[:, 1]], axis=1)
    W = np.linalg.norm(pos[faces[:, 1]] - pos[faces[:, 2]], axis=1)
    
    # exclude faces at T-Junctions
    tree = scipy.spatial.KDTree(face_centers)

    # Create a mask for surface faces (initially all True)
    is_surface = np.ones(len(faces), dtype=bool)
    
    # For each face, find nearby faces and check if they're opposite
    for i, (center, normal) in enumerate(zip(face_centers, face_normals)):
        if not is_surface[i]:
            continue
            
        # Find faces within a distance based on expected center distance in a T-junction

        tjt_center_f2f_dist = np.sqrt(L[i]**2 + W[i]**2) / 4
        # we are grabbing neighbors only for the large face at a T-junction
        neighbors = tree.query_ball_point(center, r=tjt_center_f2f_dist * 1.001)
        
        # Loop over neighbors of largest face at T-junction
        for j in neighbors:
            if i == j or not is_surface[j]:
                continue
                
            # Check if normals align
            # ideally normals should be opposite, but orientation isn't guaranteed
            if not np.abs(np.dot(normal, face_normals[j])) > 0.999:
                continue
            
            face_center_disp = face_centers[j] - center
            face_center_dist = np.linalg.norm(face_center_disp)

            # if differs from exected tjt_center_f2f_dist then it is not a T-junction
            if not (tjt_center_f2f_dist  * 0.999 < face_center_dist < tjt_center_f2f_dist  * 1.001):
                continue
            
            # get direction bw face centers
            # if direction is axis-aligned, then continue
            face_center_direction = face_center_disp / face_center_dist
            if (
                np.abs(face_center_direction[0]) > 0.999 or
                np.abs(face_center_direction[1]) > 0.999 or
                np.abs(face_center_direction[2]) > 0.999
            ):
                continue
            
            # these faces are T-junctions, not surface
            is_surface[i] = False
            is_surface[j] = False

    if debug is not None:
        print(f"omit_internal_tjunctions: omitted {np.sum(~is_surface)}/{len(faces)} faces")
    
    return faces[is_surface]

def create_surface_trimesh(pos, elems, case_name=None, debug=None):
    """
    Create a surface trimesh from a 3D hexahedral mesh
    """

    import trimesh # needs package rtree

    ###
    # extract surface nodes
    ###

    # Get all faces
    faces = make_faces(elems) # [Nfaces, 4]
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    faces = faces[unique_indices[counts == 1]]
    # Break all T-junctions
    pos_new, faces = break_tjunctions(faces, pos, debug=debug)
    # omit faces at internal T-Junctions
    faces = omit_internal_tjunctions(faces, pos_new)
    # Get unique faces. Do sorting to handle permutations
    _, unique_indices, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True, return_index=True)
    faces = faces[unique_indices[counts == 1]]
    surface_faces = faces
    
    ###
    # get surface triangles and nodes
    ###

    # surface nodes
    surface_indices = np.unique(surface_faces.reshape(-1))
    surface_pos = pos_new[surface_indices]
    
    # surface triangles
    surface_triangles = quads_to_triangles(surface_faces)
    # remap surface_triangles to only index surface nodes
    vertex_remap = np.full(len(pos_new), -1, dtype=int)
    vertex_remap[surface_indices] = np.arange(len(surface_indices))
    surface_triangles = vertex_remap[surface_triangles]

    ###
    # create trimesh object
    ###
    surface_mesh = trimesh.Trimesh(vertices=surface_pos, faces=surface_triangles)
    surface_indices = surface_indices[surface_indices < len(pos)]
    
    surface_mesh = repair_trimesh(surface_mesh, debug=debug, case_name=case_name)

    return surface_mesh, surface_indices

def repair_trimesh(mesh, debug=None, case_name=None):
    import trimesh

    mesh.remove_infinite_values()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)

    mesh.process()

    if debug is not None:
        if case_name is not None:
            print(f"\nSurface Mesh Health Check for {case_name}:")
        else:
            print("\nSurface Mesh Health Check:")

        print(f"Watertight: {mesh.is_watertight}")
        print(f"Degenerate faces: {np.sum(mesh.area_faces < 1e-6)}")
        print(f"Watertight: {mesh.is_watertight}")
        
        # Get edge face count
        if not mesh.is_watertight:
            edge_face_count = mesh.edges_unique_inverse
            unique, counts = np.unique(edge_face_count, return_counts=True)
            normal_count = np.sum(counts == 2)
            non_manifold_count = np.sum(counts > 2)
            boundary_count = np.sum(counts == 1)
            print(f"Normal edges: {normal_count}/{len(mesh.edges)}")
            print(f"Non-manifold edges: {non_manifold_count}/{len(mesh.edges)}")
            print(f"Boundary edges: {boundary_count}/{len(mesh.edges)}")

    return mesh

def save_surface_trimesh(surface_mesh, filename, edges=False):
    """
    Save a Trimesh object to VTK format with edge_face_count as a feature
    
    Args:
        surface_mesh (trimesh.Trimesh): The mesh to save
        filename (str): Output VTK filename
    """
    import pyvista as pv

    # Convert Trimesh to PyVista mesh
    pv_mesh = pv.wrap(surface_mesh)
    pv_mesh.save(filename)
    
    if edges:
        # Calculate edge_face_count
        edge_face_count = surface_mesh.edges_unique_inverse
        unique, counts = np.unique(edge_face_count, return_counts=True)

        # Create edge_face_count feature array for all edges
        edge_face_count_feature = np.zeros(len(surface_mesh.edges))
        for edge_idx, count in zip(unique, counts):
            # Assign the count to all edges with the same edges_unique_inverse value
            edge_face_count_feature[edge_face_count == edge_idx] = count
        
        # Create a PyVista PolyData object for edges
        edge_mesh = pv.PolyData()
        edge_mesh.points = surface_mesh.vertices
        edge_mesh.lines = np.column_stack(
            [np.full(len(surface_mesh.edges), 2), surface_mesh.edges]
        )
        edge_mesh['EdgeFaceCount'] = edge_face_count_feature
        edge_mesh.save(filename.replace(".vtk", "_edges.vtk"))
    
    return  

#======================================================================#
def surface_ray_intersect_trimesh(surface_mesh, ray_origins, ray_direction, debug=None):
    """
    Ray cast to surface
    """
    import trimesh # needs package rtree

    assert ray_origins.ndim == 2 and ray_origins.shape[1] == 3
    assert ray_direction.ndim == 1 and len(ray_direction) == 3

    ray_directions = np.repeat(ray_direction.reshape(1,3), len(ray_origins), axis=0)

    # get nearest point on surface
    intersect_locs, ray_indices, tri_indices = surface_mesh.ray.intersects_location(
        ray_origins,
        ray_directions,
        multiple_hits=False,
    )
    
    if len(intersect_locs) < len(ray_origins):
        if debug is not None:
            print(f"In surface_ray_intersect_trimesh:")
            print(f"num query: {len(ray_origins)}")
            print(f"num hits : {len(intersect_locs)}")
    
        intersect_locations = np.full((len(ray_origins), 3), np.nan)
        if len(intersect_locs) > 0:
            intersect_locations[ray_indices] = intersect_locs
        return intersect_locations
    else:
        return intersect_locs
    
def surface_ray_intersect_open3d(surface_mesh, ray_origins, ray_direction, debug=None):
    """
    Ray cast to surface
    """
    import open3d as o3d

    assert ray_origins.ndim == 2 and ray_origins.shape[1] == 3
    assert ray_direction.ndim == 1 and len(ray_direction) == 3

    # Convert trimesh to Open3D TriangleMesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(surface_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(surface_mesh.faces)
    
    # Create rays
    ray_directions = np.repeat(ray_direction.reshape(1,3), len(ray_origins), axis=0)
    rays = np.hstack([ray_origins, ray_directions])
    
    # Perform ray intersection
    scene = o3d.t.geometry.RaycastingScene()
    # Convert vertices to float32 and faces to uint32 before passing to Open3D
    vertices_float32 = surface_mesh.vertices.astype(np.float32)
    faces_uint32 = surface_mesh.faces.astype(np.uint32)
    scene.add_triangles(o3d.core.Tensor.from_numpy(vertices_float32),
                        o3d.core.Tensor.from_numpy(faces_uint32))
    
    ans = scene.cast_rays(o3d.core.Tensor.from_numpy(rays.astype(np.float32)))
    
    # Process results
    hit = ans['t_hit'].numpy() != np.inf
    intersect_locs = ray_origins + ray_directions * ans['t_hit'].numpy().reshape(-1,1)
    
    if np.sum(hit) < len(ray_origins):
        if debug is not None:
            print(f"In surface_ray_intersect_open3d:")
            print(f"num query: {len(ray_origins)}")
            print(f"num hits : {np.sum(hit)}")
    
        intersect_locations = np.full((len(ray_origins), 3), np.nan)
        intersect_locations[hit] = intersect_locs[hit]
        return intersect_locations
    else:
        return intersect_locs

def interpolate_missed_intersections(pos, mask, intersections, tree=None):
    """
    Fill missed intersections
    """
    if tree is None:
        tree = scipy.spatial.KDTree(pos[~mask])
    
    _, nearest_indices = tree.query(pos[mask], k=1)
    difference = pos[mask] - pos[~mask][nearest_indices]
    return intersections[nearest_indices] + difference

def dist_to_surface_in_dir(direction, surface_mesh, surface_mask, pos, tree_interior=None, debug=None):
    """
    Compute distance to surface in a given direction
    """
    direction = direction / np.linalg.norm(direction)
    
    # intersections_interior = surface_ray_intersect_trimesh(surface_mesh, pos[~surface_mask], direction, debug=debug)
    intersections_interior = surface_ray_intersect_open3d(surface_mesh, pos[~surface_mask], direction, debug=debug)

    # deal with missed intersections
    isnan_interior = np.isnan(intersections_interior.sum(axis=1))
    if np.any(isnan_interior):
        intersections_interior_nan = interpolate_missed_intersections(
            pos[~surface_mask], isnan_interior, intersections_interior[~isnan_interior])
        intersections_interior[isnan_interior] = intersections_interior_nan

    # deal with surface
    intersections_surface = interpolate_missed_intersections(
        pos, surface_mask, intersections_interior, tree=tree_interior)

    # assemble intersections array
    intersections = np.zeros((len(pos), 3,))
    intersections[~surface_mask] = intersections_interior
    intersections[surface_mask] = intersections_surface

    # compute distances
    distances = np.abs(np.dot(pos - intersections, direction))
    
    return distances

#======================================================================#
def sdf_features(
    pos, elems,
    surf_verts=None, surf_faces=None,
    case_name=None,
    debug=None,
):
    """
    Compute distance to surface
    Args:
        pos: (N, 3) array of node positions
        elems: (E, 8) array of hex element connectivity
    Returns:
        directions: (N, 3) array of direction vectors (x, y, z components)
    """

    surface_mesh, surface_indices = create_surface_trimesh(pos, elems, case_name=case_name, debug=debug)
    
    if (surf_verts is not None) and (surf_faces is not None):
        surf_mesh = trimesh.Trimesh(vertices=surf_verts, faces=surf_faces)
        surf_mesh = repair_trimesh(surf_mesh, debug=debug, case_name=case_name)
    else:
        surf_mesh = None

    # save_surface_mesh = True
    # if save_surface_mesh:
    #     save_dir = os.path.join('out', 'am', 'exp', 'surface_mesh')
    #     os.makedirs(save_dir, exist_ok=True)
    #     filename1 = os.path.join(save_dir, f'{case_name}_dev.vtk')
    #     filename2 = os.path.join(save_dir, f'{case_name}_stl.vtk')
    #     save_surface_trimesh(surface_mesh, filename1, edges=True)
    #     save_surface_trimesh(surf_mesh   , filename2, edges=True)

    ###
    # get SDF direction and magnitude
    ###
    nearest_point, _, _ = surface_mesh.nearest.on_surface(pos)
    sdf_direction = nearest_point - pos
    sdf_magnitude = np.linalg.norm(sdf_direction, axis=1)

    ###
    # get surface mask
    ###
    surface_mask = np.zeros(len(pos), dtype=bool)
    surface_mask[surface_indices] = True
    
    ###
    # misc
    ###
    tree_interior = scipy.spatial.KDTree(pos[~surface_mask])

    # get distance to surface in [-1/1, 0, 0], [0, -1/1, 0], [0, 0, -1/1]
    directions = (
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., 0., 1.]),
    )
    
    distances = [] # -(X, Y, Z), (X, Y, Z)

    for sign in [-1., 1.]:
        for direction in directions:
            direction = direction * sign
            dist = dist_to_surface_in_dir(
                direction, surface_mesh, surface_mask,
                pos, tree_interior=tree_interior, debug=debug
            ).reshape(-1,1)
            distances.append(dist)

    distances = np.hstack(distances) # [N, 6]
    
    return np.hstack([
        surface_mask.reshape(-1,1),
        sdf_direction,
        sdf_magnitude.reshape(-1,1),
        distances,
    ])

#======================================================================#
#
