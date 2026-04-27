import cv2

from .logger import log
from .utils import CALIBRATION_BOARD_IMAGE, create_board


def create_board_image() -> None:
    board, _ = create_board()
    img = board.generateImage((1280, 960))
    log.debug(f"Press any key to view the generated calibration board image...")
    cv2.imshow("Calibration Visualization", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(str(CALIBRATION_BOARD_IMAGE), img)


if __name__ == "__main__":
    create_board_image()
