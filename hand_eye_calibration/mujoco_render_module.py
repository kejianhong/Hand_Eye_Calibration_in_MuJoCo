import glfw
import mujoco
import numpy as np
from numpy.typing import NDArray

from .logger import log


class MujocoRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_id: str,
    ):
        self.model = model
        self.data = data
        self.camera_id = camera_id
        self.resolution: NDArray[np.int32] = self.model.cam_resolution[camera_id]

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.offscreen_window = None
        self.offscreen_scene = None
        self.offscreen_context = None
        self.offscreen_cam = None
        self.viewport = None
        self.rgb_buffer = None
        self.depth_buffer = None
        scene_option = mujoco.MjvOption()
        self.setup_offscreen_rendering()

    def setup_offscreen_rendering(self) -> None:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.offscreen_window = glfw.create_window(self.resolution[0], self.resolution[1], "Offscreen Camera", None, None)

        if not self.offscreen_window:
            glfw.terminate()
            raise RuntimeError("Failed to create offscreen window")

        glfw.make_context_current(self.offscreen_window)

        self.offscreen_scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.offscreen_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        self.offscreen_cam = mujoco.MjvCamera()
        self.offscreen_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        if self.camera_id != -1:
            self.offscreen_cam.fixedcamid = self.camera_id

        self.viewport = mujoco.MjrRect(0, 0, self.resolution[0], self.resolution[1])

        self.rgb_buffer = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        self.depth_buffer = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.float32)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.offscreen_context)

        mujoco.mjr_resizeOffscreen(self.resolution[0], self.resolution[1], self.offscreen_context)

        log.debug(f"Mujoco offscreen rendering environment initialized with resolution: {self.resolution}")

    def render_image(self) -> NDArray[np.uint8]:
        if not self.offscreen_window:
            raise RuntimeError("Offscreen rendering environment not initialized, please call setup_offscreen_rendering() first")

        glfw.make_context_current(self.offscreen_window)

        if self.camera_id != -1:
            mujoco.mjv_updateCamera(self.model, self.data, self.offscreen_cam, self.offscreen_scene)

        option = mujoco.MjvOption()
        perturb = mujoco.MjvPerturb()

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            option,
            perturb,
            self.offscreen_cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.offscreen_scene,
        )

        mujoco.mjr_render(self.viewport, self.offscreen_scene, self.offscreen_context)

        mujoco.mjr_readPixels(self.rgb_buffer, self.depth_buffer, self.viewport, self.offscreen_context)

        # correct: mujoco left down is (0,0)，OpenCV left up is (0,0)
        rgb_image = np.flipud(self.rgb_buffer).copy()

        return rgb_image

    def is_window_open(self) -> bool:
        if self.offscreen_window:
            return glfw.window_should_close(self.offscreen_window) == 0
        return True

    def cleanup(self) -> None:
        try:
            if self.offscreen_window:
                glfw.destroy_window(self.offscreen_window)
            glfw.terminate()
            log.debug("Mujoco rendering resources cleaned up successfully.")
        except Exception as e:
            log.error(f"Mujoco rendering resources cleanup failed: {e}")
