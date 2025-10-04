"""
Step 2: Filter noise and isolate the target object
"""
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
import os
from config import *

def remove_noise(mesh, k_neighbors=NOISE_FILTER_NEIGHBORS, std_ratio=NOISE_FILTER_STD_RATIO):
    """Statistical outlier removal"""
    print("Removing noise...")
    
    points = mesh.points
    tree = cKDTree(points)
    
    # Compute average distance to k nearest neighbors
    distances, _ = tree.query(points, k=k_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Remove outliers
    mean = mean_distances.mean()
    std = mean_distances.std()
    threshold = mean + std_ratio * std
    
    mask = mean_distances < threshold
    clean_mesh = mesh.extract_points(mask)
    
    print(f"  Removed {mesh.n_points - clean_mesh.n_points} outlier points")
    
    return clean_mesh

def smooth_mesh(mesh, iterations=SMOOTHING_ITERATIONS):
    """Smooth mesh surface"""
    print("Smoothing mesh...")
    
    smoothed = mesh.smooth(n_iter=iterations, relaxation_factor=0.1)
    
    return smoothed

def extract_largest_component(mesh):
    """Keep only the largest connected component"""
    print("Extracting largest component...")
    
    # Get connected components
    mesh = mesh.connectivity(largest=True)
    
    return mesh

def isolate_object(input_path, object_name):
    """Complete filtering and isolation pipeline"""
    print(f"\n{'='*50}")
    print(f"Filtering and Isolating: {object_name}")
    print(f"{'='*50}")
    
    # Load mesh
    mesh = pv.read(input_path)
    print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} faces")
    
    # Step 1: Remove noise
    mesh = remove_noise(mesh)
    
    # Step 2: Extract largest component
    mesh = extract_largest_component(mesh)
    
    # Step 3: Smooth
    mesh = smooth_mesh(mesh)
    
    # Step 4: Remove isolated vertices
    mesh = mesh.clean()
    
    # Save filtered mesh
    os.makedirs(FILTERED_DIR, exist_ok=True)
    output_path = f"{FILTERED_DIR}/{object_name}_clean.obj"
    mesh.save(output_path)
    
    print(f"\n✓ Clean mesh saved to {output_path}")
    print(f"  Final: {mesh.n_points} points, {mesh.n_cells} faces")
    
    return output_path

def visualize_comparison(original_path, filtered_path):
    """Show before/after comparison"""
    original = pv.read(original_path)
    filtered = pv.read(filtered_path)
    
    plotter = pv.Plotter(shape=(1, 2))
    
    # Original
    plotter.subplot(0, 0)
    plotter.add_text("Original", font_size=12)
    plotter.add_mesh(original, color='lightblue', lighting=True)
    
    # Filtered
    plotter.subplot(0, 1)
    plotter.add_text("Filtered", font_size=12)
    plotter.add_mesh(filtered, color='lightgreen', lighting=True)
    
    plotter.show()

if __name__ == "__main__":
    import sys
    
    object_name = sys.argv[1] if len(sys.argv) > 1 else TARGET_OBJECT
    input_path = f"{SCANS_DIR}/{object_name}_detected.obj"
    
    # Filter and isolate
    output_path = isolate_object(input_path, object_name)
    
    # Visualize
    visualize_comparison(input_path, output_path)