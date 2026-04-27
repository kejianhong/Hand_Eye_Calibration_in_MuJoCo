import glob
from typing import Any

import cv2
import numpy as np

from .logger import log
from .utils import CAMERA_INERPARA_FILE, DATA_PATH, create_board, create_param


def check_charuco_corners(
    board: cv2.aruco.CharucoBoard,
    dictionary: cv2.aruco.Dictionary,
    params: cv2.aruco.DetectorParameters,
    image_files: list[str],
) -> tuple[list[cv2.typing.MatLike], list[cv2.typing.MatLike], tuple[int, ...]]:
    allCharucoCorners = []
    allCharucoIds = []
    for fname in image_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if ids is not None and len(ids) > 0:
            log.info(f"Image {fname} has {len(ids)} markers, IDs: {ids.flatten().tolist()}")
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if charuco_retval and charuco_ids is not None and len(charuco_ids) >= 8:
                allCharucoCorners.append(charuco_corners)
                allCharucoIds.append(charuco_ids)
    return allCharucoCorners, allCharucoIds, img_shape


def identify_camera_inner_parameter() -> None:
    board, dictionary = create_board()
    params = create_param()
    image_files: list[str] = sorted(glob.glob(str(DATA_PATH / "*.png")))
    if not image_files:
        raise FileNotFoundError(f"Could not find any images in '{DATA_PATH}' directory.")

    allCharucoCorners, allCharucoIds, img_shape = check_charuco_corners(board, dictionary, params, image_files)
    log.info(f"There are {len(allCharucoCorners)} valid charuco corners detected from {len(image_files)} images.")

    if len(allCharucoCorners) == 0:
        raise ValueError("Not enough charuco corners detected for calibration.")

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=allCharucoCorners,
        charucoIds=allCharucoIds,
        board=board,
        imageSize=img_shape[::-1],
        cameraMatrix=None,
        distCoeffs=None,
    )  # type: ignore[call-overload]

    log.info(f"Intrinsic matrix:\n{cameraMatrix}")
    log.info(f"Distortion coefficients:\n{distCoeffs.ravel()}")
    DATA_PATH.mkdir(exist_ok=True)
    np.savez(
        CAMERA_INERPARA_FILE,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
    )


if __name__ == "__main__":
    identify_camera_inner_parameter()
