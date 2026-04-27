import glob
import os
from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from .logger import log
from .utils import CAMERA_CALIBRATION_FILE, CAMERA_INERPARA_FILE, DATA_PATH, create_board, create_param


@lru_cache
def load_camera() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    try:
        calib_data = np.load(CAMERA_INERPARA_FILE)
        camera_matrix = calib_data["cameraMatrix"]
        dist_coeffs = calib_data["distCoeffs"]
        log.debug(f"Intrinsic matrix:\n{camera_matrix}")
        log.debug(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
        return camera_matrix, dist_coeffs
    except Exception as e:
        log.error(f"Failed to load camera calibration data from {CAMERA_INERPARA_FILE}: {e}")
        raise FileNotFoundError(f"Camera calibration file not found at {CAMERA_INERPARA_FILE}")


def collect_data() -> tuple[
    list[cv2.typing.MatLike],
    list[cv2.typing.MatLike],
    list[cv2.typing.MatLike],
    list[cv2.typing.MatLike],
    list[int],
]:
    camera_matrix, dist_coeffs = load_camera()
    board, dictionary = create_board()
    params = create_param()

    image_files = sorted(glob.glob(str(DATA_PATH / "screenshot_*.png")))
    log.debug(f"Found {len(image_files)} calibration images.")

    R_target2cam_list: list[cv2.typing.MatLike] = []
    t_target2cam_list: list[cv2.typing.MatLike] = []
    R_end2base_list: list[cv2.typing.MatLike] = []
    t_end2base_list: list[cv2.typing.MatLike] = []
    valid_indices: list[int] = []

    for i, img_file in enumerate(image_files):
        log.debug(f"Processing {i+1}/{len(image_files)}: {os.path.basename(img_file)}")
        img = cv2.imread(img_file)
        assert img is not None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # method 1
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # method 2
        # gray = cv2.equalizeHist(gray)

        # method 3
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # gray = clahe.apply(gray)

        pose_file = img_file.replace("screenshot", "cartesian_pos").replace(".png", ".npy")
        assert os.path.exists(pose_file), f"{pose_file=}"

        try:
            T_end2base: NDArray[np.float64] = np.load(pose_file)
            assert T_end2base.shape == (4, 4)
            R_end2base = T_end2base[:3, :3]
            t_end2base = T_end2base[:3, 3].reshape(3, 1)

            # Check if `R_end2base` is a valid rotation matrix.
            det = np.linalg.det(R_end2base)
            if np.abs(det - 1.0) > 0.01:
                log.warning(f"The rotation matrix from file {pose_file} is not valid (det={det:.3f}). Skipping this data point.")
                continue
        except Exception as e:
            log.error(f"Error loading robot pose from {pose_file}: {e}.")

        # Check the calibration board.
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        if ids is not None and len(ids) > 0:
            log.debug(f"There are {len(ids)} ArUco markers detected in the image.")
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if charuco_retval > 0 and charuco_ids is not None and len(charuco_ids) >= 8:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)
                log.debug(f"Successfully detected {len(charuco_ids)} ChArUco corners for pose estimation.")
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charucoCorners=charuco_corners,
                    charucoIds=charuco_ids,
                    board=board,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvec=None,
                    tvec=None,
                    useExtrinsicGuess=False,
                )  # type: ignore[call-overload]

                if retval:
                    R_board2cam, _ = cv2.Rodrigues(rvec)
                    R_target2cam_list.append(R_board2cam)
                    t_target2cam_list.append(tvec.reshape(3, 1))
                    R_end2base_list.append(R_end2base)
                    t_end2base_list.append(t_end2base)
                    valid_indices.append(i)
                    log.debug(f"Estimated board pose for image {i+1}: R=\n{R_board2cam}\nt={tvec.flatten()}")
                else:
                    log.warning(f"Unable to estimate pose for image {i+1} using ChArUco corners.")
            else:
                log.warning(f"Image {i+1} does not have enough ChArUco corners.")
        else:
            log.warning(f"Image {i+1} does not have any ArUco markers.")

    log.info(f"Valid data points collected: {len(valid_indices)} out of {len(image_files)}.")

    return R_target2cam_list, t_target2cam_list, R_end2base_list, t_end2base_list, valid_indices


def calibrate() -> None:
    camera_matrix, dist_coeffs = load_camera()
    R_target2cam_list, t_target2cam_list, R_end2base_list, t_end2base_list, valid_indices = collect_data()
    assert len(R_target2cam_list) >= 3, "At least 3 valid data points are required for hand-eye calibration."

    R_target2cam_np = np.array(R_target2cam_list)
    t_target2cam_np = np.array(t_target2cam_list)
    R_gripper2base_np = np.array(R_end2base_list)
    t_gripper2base_np = np.array(t_end2base_list)
    log.debug(f"Shape of R_target2cam: {R_target2cam_np.shape}")
    log.debug(f"Shape of t_target2cam: {t_target2cam_np.shape}")
    log.debug(f"Shape of R_gripper2base: {R_gripper2base_np.shape}")
    log.debug(f"Shape of t_gripper2base: {t_gripper2base_np.shape}")

    # Different hand-eye calibration methods.
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }

    best_error = float("inf")
    best_method: str | None = None
    best_R: NDArray[np.float64] | None = None
    best_t: NDArray[np.float64] | None = None

    log.debug("Trying different hand-eye calibration methods...")
    for method_name, method_code in methods.items():
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base_np,
                t_gripper2base=t_gripper2base_np,
                R_target2cam=R_target2cam_np,
                t_target2cam=t_target2cam_np,
                method=method_code,
            )  # type: ignore[call-overload]

            errors = []
            for i in range(len(R_target2cam_np)):
                T_gripper2base = np.eye(4)
                T_gripper2base[:3, :3] = R_gripper2base_np[i]
                T_gripper2base[:3, 3] = t_gripper2base_np[i].flatten()

                T_target2cam = np.eye(4)
                T_target2cam[:3, :3] = R_target2cam_np[i]
                T_target2cam[:3, 3] = t_target2cam_np[i].flatten()

                T_cam2gripper = np.eye(4)
                T_cam2gripper[:3, :3] = R_cam2gripper
                T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

                T_target2base_i = T_gripper2base @ T_cam2gripper @ T_target2cam

                if i == 0:
                    T_base_board_ref = T_target2base_i
                else:
                    pos_error = np.linalg.norm(T_target2base_i[:3, 3] - T_base_board_ref[:3, 3])
                    errors.append(pos_error)

            avg_error = np.mean(errors) if len(errors) > 0 else 0
            log.debug(f"{method_name}: {avg_error=:.6f}")

            if avg_error < best_error:
                best_error = avg_error
                best_method = method_name
                best_R = R_cam2gripper
                best_t = t_cam2gripper

        except Exception as e:
            log.warning(f"{method_name} method failed: {e}")

    if best_R is not None and best_t is not None:
        R_cam2gripper = best_R
        t_cam2gripper = best_t
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

        T_gripper2base_0 = np.eye(4)
        T_gripper2base_0[:3, :3] = R_gripper2base_np[0]
        T_gripper2base_0[:3, 3] = t_gripper2base_np[0].flatten()
        T_target2cam_0 = np.eye(4)
        T_target2cam_0[:3, :3] = R_target2cam_np[0]
        T_target2cam_0[:3, 3] = t_target2cam_np[0].flatten()
        T_target2base = T_gripper2base_0 @ T_cam2gripper @ T_target2cam_0
        log.debug(f"Transform matrix of calibration board in base frame:\n{T_target2base}")

        t_target2base = T_target2base[:3, 3]
        rot_target2base = R.from_matrix(T_target2base[:3, :3])
        euler_angles_target2base = rot_target2base.as_euler("xyz", degrees=True)
        log.debug(f"Translation vector of target in base: {t_target2base}")
        log.debug(f"Euler angles(x,y,z) of target in base: {euler_angles_target2base} in rad, {np.degrees(euler_angles_target2base)} in degrees.")

        # Check the calibration result.
        position_errors = []
        orientation_errors = []
        for i in range(len(R_target2cam_np)):
            T_gripper2base = np.eye(4)
            T_gripper2base[:3, :3] = R_gripper2base_np[i]
            T_gripper2base[:3, 3] = t_gripper2base_np[i].flatten()

            T_target2cam = np.eye(4)
            T_target2cam[:3, :3] = R_target2cam_np[i]
            T_target2cam[:3, 3] = t_target2cam_np[i].flatten()

            T_target2base_pred = T_gripper2base @ T_cam2gripper @ T_target2cam
            log.debug(f"Image [{i}], transform matrix of target in base:\n{T_target2base_pred}")

            if i == 0:
                T_target2base_ref = T_target2base_pred
            else:
                pos_error = np.linalg.norm(T_target2base_pred[:3, 3] - T_target2base_ref[:3, 3])
                position_errors.append(pos_error)

                R_pred = T_target2base_pred[:3, :3]
                R_ref = T_target2base_ref[:3, :3]
                R_diff = R_pred @ R_ref.T
                angle_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
                orientation_errors.append(angle_error)
                log.debug(f"Image [{i}], position error = {pos_error:.6f} m, orientation error = {np.degrees(angle_error):.3f} in degree")

        if len(position_errors) > 0:
            log.debug(f"Average position error: {np.mean(position_errors):.6f} m")
            log.debug(f"Maximum position error: {np.max(position_errors):.6f} m")
            log.debug(f"Average orientation error: {np.degrees(np.mean(orientation_errors)):.3f} in degree")
            log.debug(f"Maximum orientation error: {np.degrees(np.max(orientation_errors)):.3f} in degree")

        log.info(f"{best_method} method selected with average position error: {best_error:.6f}")
        log.info(f"Rotation matrix of camera in gripper:\n{R_cam2gripper}")
        log.info(f"Translation vector of camera in gripper: {t_cam2gripper.flatten()}")
        log.info(f"Transform matrix of camera in gripper:\n{T_cam2gripper}")

        log.info(f"Saving calibration results to {CAMERA_CALIBRATION_FILE}")
        DATA_PATH.mkdir(exist_ok=True)
        np.savez(
            CAMERA_CALIBRATION_FILE,
            T_cam2gripper=T_cam2gripper,
            T_target2base=T_target2base,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            valid_indices=valid_indices,
        )

        # Visualize the calibration result on the images.
        log.info("Visualizing calibration results on the images...")
        visualize_calibration()

    else:
        log.warning("All hand-eye calibration methods failed. No valid result obtained.")


def visualize_calibration() -> None:
    camera_matrix, dist_coeffs = load_camera()
    board, dictionary = create_board()
    image_files = sorted(glob.glob(str(DATA_PATH / "screenshot_*.png")))
    for img_file in image_files:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is not None and len(ids) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if charuco_retval:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                cv2.cornerSubPix(gray, charuco_corners, (5, 5), (-1, -1), criteria)
                img = cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board,
                    camera_matrix,
                    dist_coeffs,
                    None,
                    None,
                )  # type: ignore[call-overload]

                if retval:
                    img = cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01)
                    distance = np.linalg.norm(tvec)
                    cv2.putText(img, f"Distance: {distance:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Position: {tvec.flatten()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    pose_file = img_file.replace("screenshot", "cartesian_pos").replace(".png", ".npy")
                    if os.path.exists(pose_file):
                        T_base_end = np.load(pose_file)
                        pos = T_base_end[:3, 3]
                        cv2.putText(img, f"Robot end effector pso: {pos}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Calibration Visualization", img)
        log.debug(f"Press any key to continue to the next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate()
