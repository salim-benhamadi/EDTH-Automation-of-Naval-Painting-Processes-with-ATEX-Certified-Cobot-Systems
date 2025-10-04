import sys
import numpy as np
import open3d as o3d
import os
import sklearn.neighbors as nn

FILTERED_DIR = "data\\filtered"
TARGET_OBJECT = "data\\scans\\object_scanned.obj"

def parse_obj_vertices_with_colors(path):
    """Fallback parser for OBJ lines like:
       v x y z r g b
       returns (Nx3 verts, Nx3 colors) where colors are in 0..1
    """
    verts = []
    cols = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                # expect at least 4: 'v', x, y, z, optionally r g b
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    verts.append([x, y, z])
                    if len(parts) >= 7:
                        r, g, b = float(parts[4]), float(parts[5]), float(parts[6])
                        cols.append([r, g, b])
                    else:
                        cols.append([0.5, 0.5, 0.5])
    if len(verts) == 0:
        return None, None
    return np.array(verts, dtype=float), np.array(cols, dtype=float)

def detect_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    distance_threshold = 0.05  # adjust based on your scale (meters)
    ransac_n = 3
    num_iterations = 2000

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    return plane_model, inliers

def remove_plane(pcd, inliers):
    pcd_no_plane = pcd.select_by_index(inliers, invert=True)
    print("Points after plane removal:", np.asarray(pcd_no_plane.points).shape[0])
    #o3d.visualization.draw_geometries([pcd_no_plane], window_name="After Plane Removal")
    return pcd_no_plane

def remove_plane_triangles(mesh, inliers):
    # Compute mask for triangles where all vertices are in the plane
    triangles = np.asarray(mesh.triangles)
    vertices_in_plane = set(inliers)
    mask = [not all(v in vertices_in_plane for v in tri) for tri in triangles]

    # Keep only triangles not fully on the plane
    mesh_no_plane = o3d.geometry.TriangleMesh()
    mesh_no_plane.vertices = mesh.vertices
    mesh_no_plane.triangles = o3d.utility.Vector3iVector(triangles[mask])
    if mesh.has_vertex_colors():
        mesh_no_plane.vertex_colors = mesh.vertex_colors

    mesh_no_plane.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh_no_plane], window_name="Mesh without Floor")
    # Save cleaned mesh
    o3d.io.write_triangle_mesh("mesh_no_floor.obj", mesh_no_plane)
    return mesh_no_plane

def crop_central_region(geom, radius):
    """
    Keep only points (or vertices) within `radius` of the mesh/point-cloud centroid.

    Args:
        geom: open3d.geometry.PointCloud or open3d.geometry.TriangleMesh
        radius: float, distance threshold (same units as your data, e.g., meters)

    Returns:
        A cropped copy of the same type (PointCloud or TriangleMesh)
    """
    # get points or vertices
    pts = np.asarray(geom.points if isinstance(geom, o3d.geometry.PointCloud)
                     else geom.vertices)

    center = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    mask = dists < radius

    if isinstance(geom, o3d.geometry.PointCloud):
        cropped = geom.select_by_index(np.where(mask)[0])
    else:
        # for a mesh, keep vertices within radius and associated triangles
        keep_idx = np.where(mask)[0]
        keep_set = set(keep_idx)
        tri = np.asarray(geom.triangles)
        tri_mask = [all(v in keep_set for v in t) for t in tri]
        cropped = o3d.geometry.TriangleMesh()
        cropped.vertices = o3d.utility.Vector3dVector(pts[keep_idx])
        # remap triangle indices
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        tri_filtered = [
            [old_to_new[v] for v in t] for i, t in enumerate(tri) if tri_mask[i]
        ]
        cropped.triangles = o3d.utility.Vector3iVector(np.array(tri_filtered))
        if geom.has_vertex_colors():
            cols = np.asarray(geom.vertex_colors)[keep_idx]
            cropped.vertex_colors = o3d.utility.Vector3dVector(cols)

    print(f"Kept {np.sum(mask)} of {len(pts)} points within radius {radius:.4f}")
    return cropped

def fill_holes_poisson(mesh):
    """
    Fills holes in a mesh using Poisson surface reconstruction.
    This creates a new, watertight mesh.

    Args:
        mesh: An open3d.geometry.TriangleMesh object.

    Returns:
        A tuple containing:
        - watertight_mesh (open3d.geometry.TriangleMesh): The new, watertight mesh.
        - densities (open3d.utility.DoubleVector): The density of each vertex,
          which can be used to remove low-confidence areas.
    """
    # Create a point cloud from the mesh vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    
    # Normals are crucial for Poisson reconstruction
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pcd.normals = mesh.vertex_normals
    
    print("Running Poisson surface reconstruction...")
    # The 'depth' parameter controls the resolution of the output mesh.
    # Higher depth means more detail, but also more computation and potential for noise.
    # A depth of 8 or 9 is a good starting point.
    watertight_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.2, linear_fit=False
    )
    
    print("Removing low-density vertices...")
    # The reconstruction can create spurious triangles in low-density areas.
    # We can remove these by thresholding the density.
    # A common approach is to remove vertices with a density below a certain quantile.
    densities_arr = np.asarray(densities)
    density_threshold = np.quantile(densities_arr, 0.05) # Remove the bottom 5%
    vertices_to_remove = densities_arr < density_threshold
    watertight_mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # The output mesh from Poisson is often dense and can be simplified
    # watertight_mesh = watertight_mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

    return watertight_mesh

def smooth_mesh_taubin(mesh, number_of_iterations=3):
    """
    Smoothes a mesh using Taubin filtering, which reduces shrinkage.

    Args:
        mesh: An open3d.geometry.TriangleMesh object.
        number_of_iterations (int): The number of smoothing iterations.
                                    More iterations lead to a smoother result but
                                    can also remove desired features.

    Returns:
        open3d.geometry.TriangleMesh: The smoothed mesh.
    """
    print(f"Smoothing mesh with Taubin filter ({number_of_iterations} iterations)...")
    smoothed_mesh = mesh.filter_smooth_taubin(number_of_iterations=number_of_iterations)
    
    # Smoothing can slightly alter the geometry, so it's good practice
    # to recompute normals and clean up the mesh.
    smoothed_mesh.compute_vertex_normals()
    smoothed_mesh.remove_degenerate_triangles()
    smoothed_mesh.remove_unreferenced_vertices()

    return smoothed_mesh

def isolate_largest_cluster(mesh, eps=0.05, min_points=50):
    """
    Isolates the largest cluster from a mesh using DBSCAN clustering on its vertices.

    Args:
        mesh (open3d.geometry.TriangleMesh): The input mesh, which might contain
                                              multiple disconnected objects and noise.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other. This is the most important
                     DBSCAN parameter to tune. It's in the same units as your mesh
                     (e.g., meters). A good starting point is a value slightly
                     larger than the average distance between points on your object's surface.
        min_points (int): The number of samples (or points) in a neighborhood for a
                          point to be considered as a core point. This value helps
                          filter out noise.

    Returns:
        open3d.geometry.TriangleMesh: A new mesh containing only the largest object.
                                      Returns an empty mesh if no clusters are found.
    """
    # 1. Convert mesh vertices to a point cloud for clustering
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices

    # 2. Perform DBSCAN clustering
    # The output is a vector of labels; -1 indicates noise.
    print("Clustering vertices with DBSCAN...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    # 3. Find the largest cluster
    # First, count occurrences of each label (ignoring -1 for noise)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if len(counts) == 0:
        print("Warning: No clusters found. The result is an empty mesh. Try increasing 'eps' or decreasing 'min_points'.")
        return o3d.geometry.TriangleMesh()

    # The label of the largest cluster
    largest_cluster_label = unique_labels[3]
    print(f"Found {len(unique_labels)} clusters. The largest cluster (label {largest_cluster_label}) has {counts.max()} points.")

    # 4. Create a mask to select vertices of the largest cluster
    indices_to_keep = np.where(labels == largest_cluster_label)[0]

    # 5. Create the new mesh from the selected vertices and triangles
    object_mesh = mesh.select_by_index(indices_to_keep)

    # The select_by_index method in Open3D for meshes automatically handles
    # the selection of triangles where all vertices are within the selected indices.
    # It also cleans up the mesh. However, let's add a final cleanup for good measure.
    object_mesh.remove_unreferenced_vertices()
    object_mesh.compute_vertex_normals()

    return object_mesh




def main (input_path, output_path):
    # Load your mesh
    mesh = o3d.io.read_triangle_mesh(input_path)
    mesh.compute_vertex_normals()
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    if mesh.has_vertex_colors():
        pcd.colors = mesh.vertex_colors
    plane_model, inliers = detect_plane(pcd)
    pcd = remove_plane(pcd, inliers)
    print("plane removed")
    mesh_no_plane = remove_plane_triangles(mesh, inliers)
    print("plane triangles removed")
    # Crop to central region (adjust radius as needed)
    radius = 1  # in meters, adjust based on your data scale
    mesh_cropped = crop_central_region(mesh_no_plane, radius)
    print("cropped to central region")
    #o3d.visualization.draw_geometries([mesh_cropped], window_name="Cropped Mesh")
    print("Done cropping")
    filled_mesh = fill_holes_poisson(mesh_cropped)
    print("Holes filled")
    #o3d.visualization.draw_geometries([filled_mesh], window_name="Filled Mesh")
    print("Done hole filling")
    smoothed_mesh = smooth_mesh_taubin(filled_mesh, number_of_iterations=15)
    print("Smoothed the mesh.")
    final_mesh = isolate_largest_cluster(smoothed_mesh, eps=0.02, min_points=10)
    print("Kept largest cluster.")
    o3d.visualization.draw_geometries([final_mesh], window_name="Final Smoothed Mesh")
    print("Done.")
    #save final mesh
    o3d.io.write_triangle_mesh(output_path, final_mesh)   
    print(f"Saved final mesh to {output_path}")



if __name__ == "__main__":
    object_name = sys.argv[1] if len(sys.argv) > 1 else TARGET_OBJECT
    os.makedirs(FILTERED_DIR, exist_ok=True)
    output_path = f"{FILTERED_DIR}\\{object_name}_clean.obj"
    main(object_name, output_path)