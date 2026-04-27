from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .logger import log


class OpenCVRenderer:
    def __init__(self, window_name: str = "Camera View", resolution: tuple[int, int] = (1280, 960)) -> None:
        self.window_name = window_name
        self.resolution = resolution
        self._create_window()
        log.debug(f"OpenCV {window_name} has been created with resolution: {resolution}")

    def _create_window(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.resolution[0], self.resolution[1])

    def convert_image(self, rgb_image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        return bgr_image  # type: ignore[return-value]

    def show_image(self, rgb_image: NDArray[np.uint8]) -> None:
        bgr_image = self.convert_image(rgb_image)
        cv2.imshow(self.window_name, bgr_image)
        cv2.waitKey(1)

    def capture_screenshot(self, image: NDArray[np.uint8], filename: Path) -> None:
        image = self.convert_image(image)
        cv2.imwrite(str(filename), image)
        log.debug(f"Screenshot saved as: {filename}")

    def cleanup(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
            log.debug(f"OpenCV renderer: Window '{self.window_name}' has been closed")
        except Exception as e:
            log.error(f"Error occurred while cleaning up OpenCV renderer: {e}")
