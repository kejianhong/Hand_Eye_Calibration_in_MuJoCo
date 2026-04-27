from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

DATA_PATH = Path("identify_calib_board_data")
CAMERA_INERPARA_FILE = DATA_PATH / "camera_calibration.npz"
CAMERA_CALIBRATION_FILE = DATA_PATH / "hand_eye_calibration.npz"
CALIBRATION_BOARD_IMAGE = Path("universal_robots_ur5e") / "charuco_board.png"


@lru_cache(maxsize=5)
def create_param() -> cv2.aruco.DetectorParameters:
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshConstant = 7
    params.minMarkerDistanceRate = 0.02
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5  # Must be an odd number
    params.cornerRefinementMaxIterations = 30  # Maximum number of iterations
    params.cornerRefinementMinAccuracy = 0.01  # Minimum accuracy
    return params


@lru_cache(maxsize=5)
def create_board() -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
    squares_x = 8
    squares_y = 6
    square_length = 0.03
    marker_length = 0.0225  # Usually should be 75% of the `square_length`.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=dictionary,
    )
    return board, dictionary


def transform_points(t: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Transform a Nx3 array of points by a transform matrix.
    :param t: A 4x4 transform matrix.
    :param points: A Nx3 array of points.
    :return: A Nx3 array of points after transformation.
    """
    assert t.shape == (4, 4)
    assert points.shape[1] == 3
    new_points: NDArray[np.float64] = np.dot(points, np.transpose(t[:3, :3])) + t[:3, 3]  # a * b = b' * a' when multiplying matrix and vector.
    return new_points
