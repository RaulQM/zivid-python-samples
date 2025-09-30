from pathlib import Path

import cv2
import numpy as np
import zivid
import sys
sys.path.append("/home/raul/data_processing/Zivid/zivid-python-samples/modules/zividsamples")
from display import display_bgr
from paths import get_sample_data_path
from save_load_matrix import assert_affine_matrix_and_save
import matplotlib.pyplot as plt
from glob import glob
import matplotlib
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
matplotlib.use('TkAgg')

def remove_duplicates_and_merge_with_kdtree(merged_point_cloud, cropped_source_ptc, threshold=0.01):
    """
    Efficiently removes duplicate points from cropped_source_ptc and merges them into merged_point_cloud.
    Duplicates are considered points within the specified threshold distance.
    
    Args:
        merged_point_cloud (o3d.geometry.PointCloud): The current merged point cloud.
        cropped_source_ptc (o3d.geometry.PointCloud): The new point cloud to merge.
        threshold (float): The minimum distance between points to consider them duplicates.
        
    Returns:
        o3d.geometry.PointCloud: The updated merged point cloud.
    """
    # Create a KDTree from the existing points in merged_point_cloud
    kdtree = o3d.geometry.KDTreeFlann(merged_point_cloud)
    
    # Get the points and colors from cropped_source_ptc
    new_points = np.asarray(cropped_source_ptc.points)
    new_colors = np.asarray(cropped_source_ptc.colors)
    
    # Prepare arrays to store unique points and colors
    unique_points = []
    unique_colors = []
    
    for new_point, new_color in zip(new_points, new_colors):
        # Use KDTree to find the nearest neighbor within the threshold distance
        [_, idx, _] = kdtree.search_radius_vector_3d(new_point, threshold)
        
        # If no points are within the threshold, it's a unique point
        if len(idx) == 0:
            unique_points.append(new_point)
            unique_colors.append(new_color)
    
    # If there are unique points, convert them to an Open3D PointCloud and merge
    if unique_points:
        unique_points = np.array(unique_points)
        unique_colors = np.array(unique_colors)
        
        # Create a new point cloud for the unique points
        new_point_cloud = o3d.geometry.PointCloud()
        new_point_cloud.points = o3d.utility.Vector3dVector(unique_points)
        new_point_cloud.colors = o3d.utility.Vector3dVector(unique_colors)

        # Merge the unique points and their colors into the merged point cloud
        merged_point_cloud += new_point_cloud

    return merged_point_cloud

def estimate_normals(point_cloud, voxel_size):
    point_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size, max_nn=30))

def fine_registration(source, target, initial_transformation, voxel_size):
    # Apply the initial transformation to both point clouds
    source.transform(initial_transformation)

    # Estimate normals for both point clouds (point-to-plane ICP requires this)
    estimate_normals(source, voxel_size)
    estimate_normals(target, voxel_size)

    distance_threshold = voxel_size # Reduced distance threshold for better accuracy
    
    # Define stricter convergence criteria for more accurate results
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100000, relative_fitness=1e-6, relative_rmse=1e-6)
    
    # Perform ICP with point-to-plane transformation estimation
    result_icp = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(), icp_criteria)
    
    return result_icp.transformation

def register_point_clouds(source, target, voxel_size):
    # print("Performing fine registration...")
    refined_transform = fine_registration(source, target, np.eye(4), voxel_size)

    # print("Registration completed.")
    return refined_transform

def _draw_detected_marker(bgra_image: np.ndarray, detection_result: zivid.calibration.DetectionResult) -> np.ndarray:
    """Draw detected ArUco marker on the BGRA image based on Zivid ArUco marker detection results.

    Args:
        bgra_image: The input BGRA image.
        detection_result: The result object containing detected ArUco markers with their corners.

    Returns:
        bgra_image: The BGR image with ArUco detected marker drawn on it.
    """
    bgr = bgra_image[:, :, 0:3].copy()
    marker_corners = detection_result.detected_markers()[0].corners_in_pixel_coordinates

    for i, corner in enumerate(marker_corners):
        start_point = tuple(map(int, corner))
        end_point = tuple(map(int, marker_corners[(i + 1) % len(marker_corners)]))
        cv2.line(bgr, start_point, end_point, (0, 255, 0), 2)

    return bgr

def _create_open3d_point_cloud(point_cloud: zivid.PointCloud) -> o3d.geometry.PointCloud:
    """Create a point cloud in Open3D format from NumPy array.

    Args:
        point_cloud: Zivid point cloud

    Returns:
        refined_point_cloud_open3d: Point cloud in Open3D format without Nans or non finite values

    """
    xyz = point_cloud.copy_data("xyz")
    rgba = point_cloud.copy_data("rgba")

    xyz = np.nan_to_num(xyz).reshape(-1, 3)
    rgb = rgba[:, :, 0:3].reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz.astype(np.float64)))
    point_cloud_open3d.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255)

    refined_point_cloud_open3d = o3d.geometry.PointCloud.remove_non_finite_points(
        point_cloud_open3d, remove_nan=True, remove_infinite=True
    )
    return refined_point_cloud_open3d

def _main() -> None:
    with zivid.Application():

        # Initialize a container for the merged point cloud
        merged_point_cloud = o3d.geometry.PointCloud()

        data_files = sorted(glob("/home/raul/data_processing/Zivid/zivid-python-samples/source/applications/advanced/CalibrationBoardInArucoMarkerOrigin*.zdf"))
        
        # Define voxel size for downsampling
        voxel_size = 1.0
        nb_neighbors = 20  # Number of neighbors to analyze for a point
        std_ratio = 2.0    # Threshold for standard deviation

        offx = 60
        offy = 110

        # Filtering step
        xi = -30 + offx
        xf = 210 - offx # 240
        yi = -270 + offy
        yf = 30 - offy
        zi = -3

        for keys, values in enumerate(data_files[0:]):

            print(f"Reading ZDF frame from file: {values}")
            frame = zivid.Frame(values)
            aP = frame.point_cloud()

            source_ptc = _create_open3d_point_cloud(aP)

            # Convert the merged point cloud to a NumPy array
            points = np.asarray(source_ptc.points)
            colors = np.asarray(source_ptc.colors)

            # Apply filtering based on x and y limits
            cropped_indices = (points[:, 0] >= xi) & (points[:, 0] <= xf) & (points[:, 1] >= yi) & (points[:, 1] <= yf) & (points[:, 2] <= zi)
            cropped_points = points[cropped_indices]
            cropped_colors = colors[cropped_indices]

            # Create a new Open3D point cloud with filtered data
            cropped_source_ptc = o3d.geometry.PointCloud()
            cropped_source_ptc.points = o3d.utility.Vector3dVector(cropped_points)
            cropped_source_ptc.colors = o3d.utility.Vector3dVector(cropped_colors)

            # Remove noise using Statistical Outlier Removal
            cl, ind = cropped_source_ptc.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            # Select inliers (noise-free points)
            cropped_source_ptc = cropped_source_ptc.select_by_index(ind)

            if keys > 0:
                # Register the point clouds
                transformation = register_point_clouds(cropped_source_ptc, cropped_target_ptc, voxel_size)

                # Apply the transformation to the source point cloud
                cropped_source_ptc.transform(transformation)

                merged_point_cloud = remove_duplicates_and_merge_with_kdtree(merged_point_cloud, cropped_source_ptc, threshold=0.5)

            else:
                merged_point_cloud += cropped_source_ptc

            o3d.visualization.draw_geometries([merged_point_cloud])

            cropped_target_ptc = merged_point_cloud

        # Convert the merged point cloud to a NumPy array
        points = np.asarray(merged_point_cloud.points)
        colors = np.asarray(merged_point_cloud.colors)

        # Apply filtering based on x and y limits
        cropped_indices = (points[:, 0] >= xi) & (points[:, 0] <= xf) & (points[:, 1] >= yi) & (points[:, 1] <= yf) & (points[:, 2] <= zi)
        cropped_points = points[cropped_indices]
        cropped_colors = colors[cropped_indices]

        # Create a new Open3D point cloud with filtered data
        cropped_point_cloud = o3d.geometry.PointCloud()
        cropped_point_cloud.points = o3d.utility.Vector3dVector(cropped_points)
        cropped_point_cloud.colors = o3d.utility.Vector3dVector(cropped_colors)

        # Remove noise using Statistical Outlier Removal
        print("Removing noise using Statistical Outlier Removal...")
        nb_neighbors = 20  # Number of neighbors to analyze for a point
        std_ratio = 2.0    # Threshold for standard deviation
        cl, ind = cropped_point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        # Select inliers (noise-free points)
        final_point_cloud = cropped_point_cloud.select_by_index(ind)

        # Specify the path where you want to save the final point cloud
        output_file_path = "/home/raul/data_processing/Zivid/zivid-python-samples/source/applications/advanced/point_cloud.ply"
        # Save the point cloud
        o3d.io.write_point_cloud(output_file_path, cropped_point_cloud, write_ascii=True)

        # Visualize the denoised point cloud
        print("Visualizing the denoised point cloud...")
        o3d.visualization.draw_geometries([cropped_point_cloud])

if __name__ == "__main__":
    _main()
