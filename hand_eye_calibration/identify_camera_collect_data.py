import time

import mujoco
import mujoco.viewer
import numpy as np

from .actulator_jpos_module import ActuatorControllerJpos
from .logger import log
from .mujoco_render_module import MujocoRenderer
from .opencv_render_module import OpenCVRenderer
from .utils import DATA_PATH


def generate_calibration_pose() -> None:
    model = mujoco.MjModel.from_xml_path("universal_robots_ur5e/scene_calib_board.xml")
    data = mujoco.MjData(model)

    site_id = model.site("attachment_site").id
    camera_id = model.camera("end_effector_camera").id
    board_geom_id = model.geom("calibration_board_geom").id

    actuator_controller = ActuatorControllerJpos(model, data, site_id, camera_id, board_geom_id)
    mujoco_renderer = MujocoRenderer(model, data, camera_id)
    opencv_renderer = OpenCVRenderer()

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=False) as viewer:

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_CONTACT

        # actuator_controller.debug_configuration(viewer)
        configurations = actuator_controller.generate_calibration_configuration()
        while viewer.is_running() and mujoco_renderer.is_window_open():
            for count_show, configuration in enumerate(configurations):
                configuration_difference, is_in_position = actuator_controller.ensure_in_position(configuration)
                log.info(f"Configuration {count_show}: {configuration_difference = }")
                if not is_in_position:
                    log.warning(f"Can not reach the setting {configuration=}")
                    continue
                state = actuator_controller.get_joint_states(site_id)
                end_pose = state["end_pose"]
                joint_angles = state["joint_angles"]
                rgb_image = mujoco_renderer.render_image()
                opencv_renderer.show_image(rgb_image)
                opencv_renderer.capture_screenshot(rgb_image, DATA_PATH / f"screenshot_{count_show}.png")
                DATA_PATH.mkdir(exist_ok=True)
                np.save(DATA_PATH / f"cartesian_pos_{count_show}.npy", end_pose)
                np.save(DATA_PATH / f"joint_angles_{count_show}.npy", joint_angles)

                viewer.sync()
                time.sleep(1)
            break

    opencv_renderer.cleanup()
    mujoco_renderer.cleanup()
    log.info("Finished collecting data for camera calibration.")


if __name__ == "__main__":
    generate_calibration_pose()
