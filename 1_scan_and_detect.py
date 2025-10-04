"""
Step 1: Scan object with ZED camera and detect/save it
"""
import pyvista as pv
import numpy as np
import pyzed.sl as sl
import time
import os
from config import *

def initialize_camera():
    """Initialize ZED camera with tracking and mapping"""
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera Open Error: {err}")
        return None
    
    # Enable positional tracking
    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)
    
    # Enable spatial mapping
    mapping_params = sl.SpatialMappingParameters()
    mapping_params.resolution_meter = MESH_RESOLUTION
    mapping_params.range_meter = MAPPING_RANGE
    mapping_params.map_type = sl.SPATIAL_MAP_TYPE.MESH
    mapping_params.save_texture = False
    
    zed.enable_spatial_mapping(mapping_params)
    
    return zed

def scan_object(object_name, duration=SCAN_DURATION):
    """Scan object and save mesh"""
    print(f"Starting scan of {object_name}...")
    
    zed = initialize_camera()
    if zed is None:
        return None
    
    plotter = pv.Plotter()
    plotter.add_axes()
    
    runtime_params = sl.RuntimeParameters()
    start_time = time.time()
    frame_count = 0
    last_mesh = None
    mesh_actor = None
    
    print(f"Scanning for {duration} seconds...")
    print("Move camera around the object slowly!")
    
    while time.time() - start_time < duration:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            frame_count += 1
            
            if frame_count % 30 == 0:
                remaining = duration - (time.time() - start_time)
                print(f"Time remaining: {remaining:.1f}s")
                
                zed.request_spatial_map_async()
                zed_mesh = sl.Mesh()
                
                if zed.retrieve_spatial_map_async(zed_mesh) == sl.ERROR_CODE.SUCCESS:
                    if zed_mesh.vertices.size > 0:
                        last_mesh = zed_mesh
                        
                        # Visualize
                        vertices = np.array(zed_mesh.vertices, dtype=np.float32)
                        triangles = np.array(zed_mesh.triangles, dtype=np.int32)
                        faces = np.column_stack([
                            np.full(len(triangles), 3, dtype=np.int32),
                            triangles
                        ]).ravel()
                        
                        pv_mesh = pv.PolyData(vertices, faces)
                        
                        if mesh_actor is None:
                            mesh_actor = plotter.add_mesh(pv_mesh, color='white')
                        else:
                            plotter.remove_actor(mesh_actor)
                            mesh_actor = plotter.add_mesh(pv_mesh, color='white')
                        
                        plotter.render()
    
    plotter.close()
    
    # Save mesh
    os.makedirs(SCANS_DIR, exist_ok=True)
    output_path = f"{SCANS_DIR}/{object_name}_raw.obj"
    
    if last_mesh is not None and last_mesh.vertices.size > 0:
        last_mesh.save(output_path)
        print(f"✓ Saved to {output_path}")
        print(f"  Vertices: {len(last_mesh.vertices)}, Triangles: {len(last_mesh.triangles)}")
    
    zed.disable_spatial_mapping()
    zed.disable_positional_tracking()
    zed.close()
    
    return output_path

def detect_and_crop_object(mesh_path, object_name):
    """Simple bounding box detection and cropping"""
    print(f"\nDetecting and cropping {object_name}...")
    
    mesh = pv.read(mesh_path)
    
    # Simple detection: remove points far from center
    center = mesh.center
    distances = np.linalg.norm(mesh.points - center, axis=1)
    threshold = np.percentile(distances, 75)  # Keep closest 75% of points
    
    # Create mask
    mask = distances < threshold
    
    # Extract object (simplified - in reality you'd use more sophisticated methods)
    cropped_mesh = mesh.extract_points(mask)
    
    # Save detected object
    output_path = f"{SCANS_DIR}/{object_name}_detected.obj"
    cropped_mesh.save(output_path)
    
    print(f"✓ Detected object saved to {output_path}")
    print(f"  Original: {mesh.n_points} points → Detected: {cropped_mesh.n_points} points")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    object_name = sys.argv[1] if len(sys.argv) > 1 else TARGET_OBJECT
    
    # Scan
    mesh_path = scan_object(object_name)
    
    # Detect and crop
    if mesh_path:
        detect_and_crop_object(mesh_path, object_name)