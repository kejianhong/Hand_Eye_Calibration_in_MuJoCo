from .create_maker_board import create_board_image
from .handeye_claib import calibrate
from .identify_camera_collect_data import generate_calibration_pose
from .identify_camera_inerpara import identify_camera_inner_parameter
from .logger import log

if __name__ == "__main__":
    log.info("Starting the calibration process...")
    create_board_image()
    generate_calibration_pose()
    identify_camera_inner_parameter()
    calibrate()
