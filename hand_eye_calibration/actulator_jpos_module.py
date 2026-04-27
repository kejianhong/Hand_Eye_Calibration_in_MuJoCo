import glob
import time
from itertools import product
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from .logger import log
from .utils import DATA_PATH, transform_points


class ActuatorControllerJpos:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_id: str,
        camera_id: str,
        board_geom_id: str,
    ) -> None:
        self.model = model
        self.data = data
        self.site_id = site_id
        self.camera_id = camera_id
        self.board_geom_id = board_geom_id
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.joint_names = [self.model.joint(i).name for i in range(self.model.nu)]
        self.width, self.height, self.fx, self.fy, self.cx, self.cy = self._calculate_camera_intrinsics()
        self.limits = self._get_joint_limit()

    def _update_configuration(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        "Directly update the configuration."
        assert len(position) == self.model.nu
        for i, joint_name in enumerate(self.joint_names[:6]):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            pos_idx = self.model.jnt_qposadr[joint_id]
            self.data.qpos[pos_idx] = position[i]
            self.data.qvel[pos_idx] = 0  # reset the velocity
        for i, actuator_name in enumerate(self.actuator_names[:6]):
            actuator_id = self.model.actuator(actuator_name).id
            self.data.ctrl[actuator_id] = position[i]
        # Not `mj_step` because we dont care about the trajectory between two configurations.
        # The trajectory between two configurations should be computed by RRT or other motion planning algorithms.
        mujoco.mj_forward(self.model, self.data)

        # Camera pose.
        cam_pos = self.data.cam_xpos[self.camera_id].copy()
        cam_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cam_quat, self.data.cam_xmat[self.camera_id].copy())
        cam_pose = np.hstack([cam_quat, cam_pos])
        return cam_pose

    def _calculate_camera_intrinsics(self) -> tuple[int, int, float, float, float, float]:
        width, height = self.model.cam_resolution[self.camera_id]
        fovy = np.deg2rad(self.model.cam_fovy[self.camera_id])
        fx = fy = height / (2 * np.tan(fovy / 2))
        cx = width / 2
        cy = height / 2

        return width, height, fx, fy, cx, cy

    def _check_quat_difference(self, quat_threshold: float = np.deg2rad(30)) -> bool:
        cam_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cam_quat, self.data.cam_xmat[self.camera_id].copy())
        board_quat = np.zeros(4)
        mujoco.mju_mat2Quat(board_quat, self.data.geom_xmat[self.board_geom_id].copy())
        # ref https://math.stackexchange.com/questions/90081/quaternion-distance
        quat_difference = np.abs(np.arccos(2 * np.dot(cam_quat, board_quat) ** 2 - 1))
        log.debug(f"{quat_difference=}")
        return bool(quat_difference < quat_threshold)

    def _check_calibration_board_in_frustum(
        self,
        lower_plane: float = -0.7,
        upper_plane: float = -0.25,
    ) -> np.bool_:
        # Camera id.
        cam_pos = self.data.cam_xpos[self.camera_id].copy()
        cam_rot_mat = self.data.cam_xmat[self.camera_id].reshape(3, 3).copy()
        cam_pose = np.identity(4)
        cam_pose[0:3, 3] = cam_pos
        cam_pose[0:3, 0:3] = cam_rot_mat

        # Calibration board geometry id.
        board_geom_pos = self.data.geom_xpos[self.board_geom_id].copy()
        board_geom_rot_mat = self.data.geom_xmat[self.board_geom_id].reshape(3, 3).copy()
        board_geom_pose = np.identity(4)
        board_geom_pose[0:3, 3] = board_geom_pos
        board_geom_pose[0:3, 0:3] = board_geom_rot_mat

        # Calibration board corners.
        board_size = self.model.geom_size[self.board_geom_id]
        sx, sy, sz = board_size
        corner_points_in_geom = np.array(
            list(
                product(
                    [sx, -sx],
                    [sy, -sy],
                    [sz, -sz],
                )
            )
        )
        geom_in_camera = np.dot(np.linalg.inv(cam_pose), board_geom_pose)
        corner_points_in_camera = transform_points(geom_in_camera, corner_points_in_geom)
        # Check calibration board in frustum.
        # u = fx * X/Z + cx
        # v = fy * Y/Z + cy
        # where X, Y and Z are the coordinate in camera coordinate
        # X/Z = (u - cx) / fx
        # Y/Z = (v - cy) / fy
        log.debug(f"minZ={np.min(corner_points_in_camera[:, 2])}, maxZ={np.max(corner_points_in_camera[:, 2])}")
        corner_points_in_camera[:, 0] /= corner_points_in_camera[:, 2]
        corner_points_in_camera[:, 1] /= corner_points_in_camera[:, 2]
        return (
            np.min(corner_points_in_camera[:, 2]) >= lower_plane
            and np.max(corner_points_in_camera[:, 2]) <= upper_plane
            and np.all(-self.cx / self.fx < corner_points_in_camera[:, 0])
            and np.all((self.width - self.cx) / self.fx > corner_points_in_camera[:, 0])
            and np.all(-self.cy / self.fy < corner_points_in_camera[:, 1])
            and np.all((self.height-self.cy) / self.fy > corner_points_in_camera[:, 1])
        ) # fmt: skip

    def _geomId2bodyId(self, geom_id: int) -> tuple[str, str]:
        geom_name: str = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_id = self.model.geom_bodyid[geom_id]
        body_name: str = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return body_name, geom_name

    def _check_collision(self) -> bool:
        contact_dist = self.data.contact.dist
        if contact_dist.size > 0:
            for i, dist in enumerate(contact_dist):
                if dist < 0:  # only the distance < 0 means collision.
                    geom1_id, geom2_id = self.data.contact.geom[i]
                    log.debug(f"Collision detected: [{self._geomId2bodyId(geom1_id)}] x [{self._geomId2bodyId(geom2_id)}], dist={dist}")
                    return True
        return False

    def _check_joint_limit(self, configuration: NDArray[np.float64]) -> np.bool_:
        joint_limit = self._get_joint_limit()
        return np.all(joint_limit[:, 0] < configuration) and np.all(joint_limit[:, 1] > configuration)

    def get_joint_states(self, site_id: str) -> dict[str, Any]:
        joint_angles = self.data.qpos[:6].copy()
        actual_joint_torques = self.data.qfrc_actuator[:6].copy()
        joint_velocities = self.data.qvel[:6].copy()
        # Site id.
        end_pos = self.data.site_xpos[site_id].copy()
        end_rot_mat = self.data.site_xmat[site_id].reshape(3, 3).copy()
        end_pose = np.identity(4)
        end_pose[0:3, 3] = end_pos
        end_pose[0:3, 0:3] = end_rot_mat
        state = {
            "joint_angles": joint_angles,
            "joint_torques": actual_joint_torques,
            "joint_velocities": joint_velocities,
            "end_pose": end_pose,
        }
        return state

    def _get_joint_limit(self) -> NDArray[np.float64]:
        jnt_range: NDArray[np.float64] = self.model.jnt_range
        return jnt_range

    def _generate_random_configuration(self) -> NDArray[np.float64]:
        """
        Generate the random dof of the robot that satisfies the joint limit, collision check and frustum constraint.
        """
        while True:
            configuration = np.array([np.random.uniform(*limits) for limits in self._get_joint_limit()])
            try:
                self._update_configuration(configuration)
                assert not self._check_collision()
                assert self._check_calibration_board_in_frustum()
                assert self._check_quat_difference()
            except AssertionError:
                continue
            return configuration

    def _initialize_population(self, num: int = 20) -> NDArray[np.float64]:
        configurations = []
        for _ in range(num):
            configuration = self._generate_random_configuration()
            configurations.append(configuration)
        log.info(f"Generated configuration=\n{configurations}")
        return np.vstack(configurations)

    def _compute_score(self, configurations: NDArray[np.float64]) -> np.float64:
        score = []
        cam_poses = []
        for configuration in configurations:
            cam_pose = self._update_configuration(configuration)
            assert self._check_joint_limit(configuration)
            assert not self._check_collision()
            assert self._check_calibration_board_in_frustum()
            assert self._check_quat_difference()
            cam_poses.append(cam_pose)

        for i in range(1, configurations.shape[0]):
            quat1 = cam_poses[i][:4]
            trans1 = cam_poses[i][4:]
            for j in range(i):
                quat2 = cam_poses[j][:4]
                trans2 = cam_poses[j][4:]
                disQuat = np.abs(np.arccos(2 * np.dot(quat1, quat2) ** 2 - 1))  # ref https://math.stackexchange.com/questions/90081/quaternion-distance
                disTrans = np.linalg.norm(trans1 - trans2)
                disQuat = disQuat / np.pi
                disTrans = disTrans / (np.linalg.norm(trans1) + np.linalg.norm(trans2))
                score.append(disQuat + disTrans)

        return np.dot(sorted(score), np.array(list(range(len(score)))) / float(len(score)) * np.exp(-np.array(sorted(score)) * np.array(list(range(len(score)))) / len(score)))  # lambda*exp(-lambda*x)

    def generate_calibration_configuration(self, max_step: int = 10000) -> NDArray[np.float64]:
        configurations = self._initialize_population()
        score, step_size, step = self._compute_score(configurations), np.full(configurations.shape, 0.05), 0
        log.info(f"Initial {score=}")
        for cur_step in range(max_step):
            configurations_ = configurations.copy()
            delta = np.random.normal(0, step_size)
            configurations_ += delta
            step += 1
            try:
                score_ = self._compute_score(configurations_)
            except AssertionError:
                step_size *= 0.9
                log.info(f"{cur_step=}, max_step_size={np.max(step_size)}")
                if (max_step_size := np.max(step_size)) and max_step_size < 1e-4:
                    log.info(f"{max_step_size=:.6f}, stop iteration.")
                    break
            else:
                if score_ > score:
                    score = score_
                    configurations = configurations_
                    step_size = (step_size * 9 + np.abs(delta) * 2) / 10
                    log.info(f"Score {score} -> {score_}")

        log.info(f"{configurations=}")
        return configurations

    def ensure_in_position(
        self,
        configuration: NDArray[np.float64],
        max_step: int = 500,
        threshold: float = 1e-2,
    ) -> tuple[float, bool]:
        self._update_configuration(configuration)
        joint_angles: NDArray[np.float64] = np.zeros(self.model.nu)
        total_step = 0
        configuration_difference = 0.0
        is_in_position: bool = False
        while total_step < max_step:
            for _ in range(50):
                mujoco.mj_step(self.model, self.data)
            state = self.get_joint_states(self.site_id)
            joint_angles = state["joint_angles"]
            joint_torques = state["joint_torques"]
            total_step += 50
            configuration_difference = np.max(np.abs(joint_angles - configuration))
            if configuration_difference < threshold:
                is_in_position = True
                break
        log.debug(f"{is_in_position=}\ndiff={np.abs(joint_angles-configuration)}\n{joint_torques=}")
        return configuration_difference, is_in_position

    def debug_configuration(self, viewer: mujoco.viewer.Handle) -> None:
        joint_position_files = sorted(glob.glob(str(DATA_PATH / "joint_angles_*.npy")))
        for index, joint_position_file in enumerate(joint_position_files):
            configuration = np.load(joint_position_file)
            configuration_difference, is_in_position = self.ensure_in_position(configuration)
            log.info(f"{index = }, {configuration_difference = }")
            viewer.sync()
            time.sleep(1)
            from IPython import embed

            embed()  # type: ignore[no-untyped-call]
