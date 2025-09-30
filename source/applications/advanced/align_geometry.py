# Main

# Import libraries
import os
import numpy as np
from scipy.spatial import KDTree
import trimesh
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
import random
import copy
from sklearn import linear_model
import cv2
import matplotlib
import teaserpp_python
from scipy.spatial import cKDTree
import pymeshlab
import copy
import argparse
import sys
from copy import deepcopy

# pymeshlab.print_pymeshlab_version()
# filters = pymeshlab.filter_list()
# print(filters)

matplotlib.use('TkAgg')

# Path to bop_toolkit which contains the Python renderer.
bop_toolkit_path = '/home/raul/bop_toolkit'
sys.path.append(bop_toolkit_path)
from bop_toolkit_lib import inout

#Functions

def process_mesh(filename0):
    # Load the point cloud and texture using Open3D (change the filenames to your files)
    mesh = trimesh.load(filename0, force='mesh')

    # Scale mesh from mm to m
    mesh.vertices *= 0.001

    # Ensure that the frame is in the center (do not change orientation)
    mesh_center = np.mean(mesh.vertices, axis=0)
    translation_vector = -mesh_center
    mesh.vertices += translation_vector

    # #------------------------------------------------------------------------------------------------
    # # Compute the covariance matrix
    # covariance_matrix = np.cov(mesh.vertices.T)

    # # Perform SVD
    # U, S, Vt = np.linalg.svd(covariance_matrix)

    # rotation_matrix = Vt.T

    # # special reflection case
    # if np.linalg.det(rotation_matrix) < 0:
    #     rotation_matrix = -rotation_matrix

    # # Apply the rotation matrix to the mesh vertices
    # mesh.vertices = (rotation_matrix @ mesh.vertices.T).T
    # #------------------------------------------------------------------------------------------------

    # Define the rotation angle in radians and the rotation axis (e.g., [1.0, 0.0, 0.0] for x-axis)
    angle = np.deg2rad(-90)
    rotation_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Create a rotation object using SciPy's Rotation class
    rotation = Rot.from_rotvec(angle * rotation_axis)
    rotation_matrix = rotation.as_matrix()

    # Apply the rotation matrix to the mesh vertices
    mesh.vertices = (rotation_matrix @ mesh.vertices.T).T

    # Define the rotation angle in radians and the rotation axis (e.g., [1.0, 0.0, 0.0] for x-axis)
    angle = np.deg2rad(180)
    rotation_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Create a rotation object using SciPy's Rotation class
    rotation = Rot.from_rotvec(angle * rotation_axis)
    rotation_matrix = rotation.as_matrix()

    # Apply the rotation matrix to the mesh vertices
    mesh.vertices = (rotation_matrix @ mesh.vertices.T).T

    mesh.show()

    # Store the processed mesh
    save_name = os.path.join(os.path.dirname(filename0), 'final_mesh.obj')
    mesh.export(save_name, file_type='obj')
    return mesh, save_name

def mesh2ptc(trimesh_data, save_name):
    # Get the mesh vertices
    vertices = trimesh_data.vertices

    # Create o3d point cloud
    point_cloud = o3d.geometry.PointCloud()

    # Set point cloud vertices and colors
    point_cloud.points = o3d.utility.Vector3dVector(vertices)

    # Visualize the point cloud (optional)
    # o3d.visualization.draw_geometries([point_cloud])

    ptc_filename = os.path.join(os.path.dirname(save_name), os.path.basename(save_name)[:-4] + '_ptc.ply')
    # Save the point cloud to the specified file
    o3d.io.write_point_cloud(ptc_filename, point_cloud, write_ascii=True)
    return point_cloud, ptc_filename

def store_mesh(meshname):
    # Define pymeshlab object 
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(meshname)
    # save mesh
    processed_mesh_filename = os.path.join(os.path.dirname(meshname), 'final_mesh.ply')
    ms.save_current_mesh(processed_mesh_filename, binary=False, save_vertex_normal=True, save_wedge_texcoord=False, save_face_color=False) 
    return processed_mesh_filename

# Code
def main():

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptc', type=str, default='/home/raul/data_processing/Zivid/zivid-python-samples/source/applications/advanced/mesh.ply')
    args = parser.parse_args()

    trimesh_data, filename  = process_mesh(args.ptc)

    processed_mesh_filename = store_mesh(filename)

    ptc_source, ptc_filename = mesh2ptc(trimesh_data, filename)

if __name__ == '__main__':
    main()

input('Done.')
