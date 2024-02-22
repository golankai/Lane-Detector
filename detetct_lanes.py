"""
A script to detect lanes in a video of a highway drive, captured from a dashcam.
Input is a video file, output is a video file with the detected lanes drawn on it.
"""

import numpy as np
import cv2

from LaneDetector import LaneDetector

switch_args = {
    "ROI": {
        "bottom_left": (170, 930),
        "bottom_right": (1750, 930),
        "top_left": (850, 620),
        "top_right": (1100, 620),
    },
    "R_STEP": 1,
    "THETA_STEP": np.pi / 180,
    "HOUGH_TH": 50,
    "THICKNESS": 12,
    "CANNY_TH": (120, 200),
    "M_TH": (0.3, 1.7),
    "LINES_Y": (920, 620),
    "FILL_TH": 3,
    "CHANGE_LENGTH": 45,
    "MIN_LINE_LENGTH": 40,
    "MAX_LINE_GAP": 30,
    "NIGHT_MODE": False,
    "CURVE": False,
    "CROSSWALK": False,
}

night_args = {
    "ROI": {
        "bottom_left": (50, 1030),
        "bottom_right": (1810, 1030),
        "top_left": (680, 730),
        "top_right": (860, 730),
    },
    "R_STEP": 1,
    "THETA_STEP": np.pi / 180,
    "HOUGH_TH": 50,
    "THICKNESS": 12,
    "CANNY_TH": (50, 150),
    "LINES_Y": (1030, 720),
    "FILL_TH": 5,
    "M_TH": (0.3, 1.9),
    "CHANGE_LENGTH": 10,
    "MIN_LINE_LENGTH": 40,
    "MAX_LINE_GAP": 50,
    "NIGHT_MODE": True,
    "CURVE": False,
    "CROSSWALK": False,
}

crosswalk_args = {
    "ROI": {
        "bottom_left": (440, 1070),
        "bottom_right": (1770, 1070),
        "top_left": (970, 650),
        "top_right": (1020, 650),
    },
    "R_STEP": 1,
    "THETA_STEP": np.pi / 180,
    "HOUGH_TH": 20,
    "THICKNESS": 12,
    "CANNY_TH": (50, 150),
    "LINES_Y": (1079, 700),
    "FILL_TH": 3,
    "M_TH": (0.3, 1.9),
    "CHANGE_LENGTH": 10,
    "MIN_LINE_LENGTH": 200,
    "MAX_LINE_GAP": 5,
    "NIGHT_MODE": False,
    "CURVE": False,
    "CROSSWALK": True,
}

turn_args = {
    "ROI": {
        "bottom_left": (600, 1030),
        "bottom_right": (1610, 1030),
        "top_left": (880, 730),
        "top_right": (910, 730),
    },
    "R_STEP": 1,
    "THETA_STEP": np.pi / 180,
    "HOUGH_TH": 50,
    "THICKNESS": 12,
    "CANNY_TH": (50, 150),
    "LINES_Y": (1070, 780),
    "FILL_TH": 15,
    "M_TH": (0.3, 1.9),
    "CHANGE_LENGTH": 10,
    "MIN_LINE_LENGTH": 40,
    "MAX_LINE_GAP": 50,
    "NIGHT_MODE": False,
    "CURVE": True,
    "CROSSWALK": False,
}


def set_up(width: int, height: int) -> None:
    """
    Set up the figure size.
    Input:
        width: the width of the figure
        height: the height of the figure

    """
    # Set the desired figure size
    figure_width = width
    figure_height = height

    # Create a resizable window
    cv2.namedWindow("Resized Video", cv2.WINDOW_NORMAL)

    # Set the size of the window
    cv2.resizeWindow("Resized Video", figure_width, figure_height)


def _get_args(video_name):
    if video_name == "switch":
        args = switch_args
    elif video_name == "night":
        args = night_args
    elif "crosswalk" in video_name:
        args = crosswalk_args
    elif "turn" in video_name:
        args = turn_args
    else:
        args = None
    return args


def _get_lane_detector(cap, args):
    # Initialize the lane detector

    lane_detector = LaneDetector(
        cap=cap,
        roi=args["ROI"],
        r_step=args["R_STEP"],
        theta_step=args["THETA_STEP"],
        hough_th=args["HOUGH_TH"],
        thickness=args["THICKNESS"],
        canny_th=args["CANNY_TH"],
        lines_y=args["LINES_Y"],
        fill_th=args["FILL_TH"],
        m_th=args["M_TH"],
        change_length=args["CHANGE_LENGTH"],
        min_line_length=args["MIN_LINE_LENGTH"],
        max_line_gap=args["MAX_LINE_GAP"],
        night_mode=args["NIGHT_MODE"],
        curve=args["CURVE"],
        crosswalk=args["CROSSWALK"],
    )

    return lane_detector


def main():
    # Set up the figure size
    set_up(800, 500)

    # Read video file
    video_path = "src/turn.mp4"
    video_name = video_path.split("/")[-1].split(".")[0]

    cap = cv2.VideoCapture(video_path)
    args = _get_args(video_name)
    detector = _get_lane_detector(cap, args)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define elements for saving the video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = f"results/{video_name}_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stop = False

    # Process video frame by frame and display the result
    while cap.isOpened():
        if not stop:
            ret, frame = cap.read()
            if ret:
                frame = detector.process_frame(frame)

                # Save the frame
                out.write(frame)

                # Display the frame
                cv2.imshow("Resized Video", frame)
            else:
                break

        key = cv2.waitKey(2) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            stop = not stop

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
