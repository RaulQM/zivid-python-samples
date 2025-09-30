import numpy as np
from glob import glob
from os import path as osp
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import argparse
import scipy.io
import open3d as o3d
import matplotlib
from plyfile import PlyData, PlyElement
import yaml
import pymeshlab
import pyvista as pv

matplotlib.use('TkAgg')  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptc', type=str, default='/home/raul/data_processing/Zivid/zivid-python-samples/source/applications/advanced/point_cloud.ply')
    args = parser.parse_args()
    
    # Load the point cloud into a MeshSet
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.ptc)

    # Remove point cloud outliers
    ms.compute_selection_point_cloud_outliers(propthreshold=0.8, knearest=256)  # Remove outliers
    ms.meshing_remove_selected_vertices()

    sample_num = 15000  # Target number of faces

    # Downsample the point cloud
    ms.generate_simplified_point_cloud(samplenum=sample_num, bestsampleflag=True, exactnumflag=True)

    # Compute normals for surface reconstruction
    ms.compute_normal_for_point_clouds(k=30)  # Compute normals using 30 nearest neighbors
    ms.apply_normal_point_cloud_smoothing(k=30)
    
    # Get min/max Z from the point cloud
    z_vals_pcd = [v[2] for v in ms.current_mesh().vertex_matrix()]
    z_min_pcd, z_max_pcd = min(z_vals_pcd), max(z_vals_pcd)

    # Create mesh
    ms.generate_surface_reconstruction_screened_poisson(depth=9, pointweight=1.0, preclean=True)
    # ms.generate_surface_reconstruction_ball_pivoting()
    # ms.generate_surface_reconstruction_vcg()
    
    # Select vertices outside the Z range using a condition
    condition = f"(z < {z_min_pcd}) || (z > {z_max_pcd})"
    ms.compute_selection_by_condition_per_vertex(condselect=condition)
    # Remove selected vertices and their associated faces
    ms.meshing_remove_selected_vertices()
    ms.meshing_remove_unreferenced_vertices()

    # Save mesh
    direc = os.path.dirname(args.ptc)
    ms.save_current_mesh(f'{direc}/mesh_init.ply', binary=False, save_vertex_normal=True, save_wedge_texcoord=False, save_face_color=False, save_vertex_color=False)

    # Smooth mesh
    ms.apply_coord_hc_laplacian_smoothing()  # Surface-preserving Laplacian smoothing
    ms.apply_coord_taubin_smoothing(lambda_=0.5, mu=-0.53, stepsmoothnum=10)  # Taubin smoothing
    ms.compute_normal_per_face()
    ms.apply_normal_smoothing_per_face()  # Face normal smoothing

    # Add Two-Step Smoothing (not necessary!)
    # ms.apply_coord_two_steps_smoothing(stepsmoothnum=3, normalthr=60, stepnormalnum=20, stepfitnum=20)
    # ms.apply_coord_laplacian_smoothing_surface_preserving(iterations = 10)
    # ms.apply_coord_laplacian_smoothing(stepsmoothnum = 3)

    # Decimate mesh
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=sample_num)  # Simplify the mesh

    # Clean mesh
    ms.meshing_remove_connected_component_by_diameter(removeunref=True)
    ms.meshing_remove_connected_component_by_face_number(mincomponentsize=25, removeunref=True)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_folded_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_t_vertices(method='Edge Collapse', threshold=40, repeat=True)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_vertices_by_scalar()

    ms.compute_selection_bad_faces()
    ms.meshing_remove_selected_faces()
    ms.compute_selection_by_non_manifold_edges_per_face()
    ms.meshing_remove_selected_faces()
    ms.compute_selection_by_small_disconnected_components_per_face()
    ms.meshing_remove_selected_faces()
    ms.compute_selection_by_non_manifold_per_vertex()
    ms.meshing_remove_selected_vertices()

    # Repair mesh
    ms.meshing_repair_non_manifold_edges(method='Remove Faces')  
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    ms.meshing_snap_mismatched_borders(edgedistratio=0.01, unifyvertices=True)

    # Remeshing
    ms.meshing_isotropic_explicit_remeshing()  # Regularize the mesh with isotropic remeshing

    # Close bottom hole
    ms.meshing_close_holes(maxholesize=150, refinehole=True)

    # Recompute normals and smooth
    ms.compute_normal_per_vertex(weightmode='By Angle')  # Recompute normals after cleaning
    ms.apply_scalar_smoothing_per_vertex()  # Remove if scalar field smoothing is not needed

    # Save mesh
    direc = os.path.dirname(args.ptc)
    ms.save_current_mesh(f'{direc}/mesh.ply', binary=False, save_vertex_normal=True, save_wedge_texcoord=False, save_face_color=False, save_vertex_color=False)

    # Visualize the mesh
    mesh = o3d.io.read_triangle_mesh(f'{direc}/mesh.ply')
    o3d.visualization.draw_geometries([mesh])

    print("Mesh generation complete!")
