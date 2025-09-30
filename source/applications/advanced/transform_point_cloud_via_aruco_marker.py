"""
Transform a point cloud from camera to ArUco marker coordinate frame by estimating the marker's pose from the point cloud.

The ZDF file for this sample can be found under the main instructions for Zivid samples.
"""

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

import matplotlib

matplotlib.use('TkAgg')

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


def _main() -> None:
    with zivid.Application():
    
        file_nb = 3

        data_file = f"/home/raul/data_processing/Zivid/zivid-python-samples/source/applications/advanced/CalibrationBoardInCameraOrigin{file_nb}.zdf"
        print(f"Reading ZDF frame from file: {data_file}")
        frame = zivid.Frame(data_file)
        point_cloud = frame.point_cloud()

        rgba = frame.point_cloud().copy_data("rgba")  # Get point colors as [Height,Width,4] uint8 array

        print("Configuring ArUco marker")
        marker_dictionary = zivid.calibration.MarkerDictionary.aruco4x4_50
        marker_id = [1]

        print("Detecting ArUco marker")
        detection_result = zivid.calibration.detect_markers(frame, marker_id, marker_dictionary)

        if not detection_result.valid():
            raise RuntimeError("No ArUco markers detected")

        print("Converting to OpenCV image format")
        bgra_image = point_cloud.copy_data("bgra")
        bgr = _draw_detected_marker(bgra_image, detection_result)

        bgr_image_file = "ArucoMarkerDetected3.png"
        print(f"Saving 2D color image with detected ArUco marker to file: {bgr_image_file}")
        cv2.imwrite(bgr_image_file, bgr)

        transform_camera_to_marker = detection_result.detected_markers()[0].pose.to_matrix()
        print("ArUco marker pose in camera frame:")
        print(transform_camera_to_marker)
        print("Camera pose in ArUco marker frame:")
        transform_marker_to_camera = np.linalg.inv(transform_camera_to_marker)
        print(transform_marker_to_camera)

        transform_file = Path("ArUcoMarkerToCameraTransform3.yaml")
        print("Saving a YAML file with Inverted ArUco marker pose to file: ")
        assert_affine_matrix_and_save(transform_marker_to_camera, transform_file)

        aruco_marker_transformed_file = f"CalibrationBoardInArucoMarkerOrigin{file_nb}.zdf"
        print(f"Saving transformed point cloud to file: {aruco_marker_transformed_file}")
        frame.save(aruco_marker_transformed_file)


if __name__ == "__main__":
    _main()
