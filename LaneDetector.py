"""
A Lane Detector class to detect lanes in a video of a highway drive, captured from a dashcam.
"""

import numpy as np
import cv2

class LaneDetector:
    def __init__(
        self,
        cap,
        roi: dict,
        r_step: int,
        theta_step: float,
        hough_th: int,
        thickness: int,
        canny_th: tuple[int, int],
        lines_y: tuple[int, int],
        fill_th: int,
        m_th: int,
        change_length: int,
        min_line_length,
        max_line_gap: int,
        night_mode=False,
        curve=False,
        crosswalk=False,
    ):
        """
        Initialize the LaneDetector object.
        Input:
            cap: the video capture object
            roi: a dictionary of the region of interest
            r_step: the step size for the Hough transform
            theta_step: the step size for the Hough transform
            hough_th: the threshold for the Hough transform
            thickness: the thickness of the lines to be drawn
            canny_th: the thresholds for the Canny edge detector
            lines_y: the y coordinates of the top and bottom of the lines
            fill_th: the threshold for detecting a change in lanes
            m_th: the threshold for the slopes of the lines
            change_length: number of frames to show the switching lanes sign
            min_line_length: the minimum length of a line
            max_line_gap: the maximum gap between segments to be considered as a single line
        """
        # Set the parameters
        self.cap = cap
        self.ROI = roi
        self.R_STEP = r_step
        self.THETA_STEP = theta_step
        self.HOUGH_TH = hough_th
        self.THICKNESS = thickness
        self.CANNY_TH = canny_th
        self.LINES_Y = lines_y
        self.FILL_TH = fill_th
        self.M_TH = m_th
        self.CHANGE_LENGTH = change_length
        self.MIN_LINE_LENGTH = min_line_length
        self.MAX_LINE_GAP = max_line_gap
        self.NIGHT_MODE = night_mode
        self.CURVE = curve
        self.CROSSWALK = crosswalk
        self.ROI_CURVE = {
        "bottom_left": (840, 780),
        "bottom_right": (1070, 780),
        "top_left": (945, 650),
        "top_right": (1000, 650),
        }
        
        self.latest_curve_x = []

        # Initialize some internal variables
        self.frames_wo_lanes = 0
        self.change_lane = -1

        self.last_left = self.last_right = None

        self.left_slopes = [0]
        self.right_slopes = [0]

        self.num_of_windows = 22
        self.curve_slopes = self.num_of_windows * [0]
        self.window_height = 18

        if self.CROSSWALK:
            self.max_area = self.max_area_orig = 29000

    def process_frame(self, lane_im: np.ndarray) -> np.ndarray:
        """
        Process a single frame of a video to detect lanes.
        Input:
            frame: a single frame of a video
        Output:
            frame: the same frame with detected lanes drawn on it
            or other needed output instead
        """
        orig_frame = lane_im.copy()
        # If we are in the middle of a lane change, show the sign
        if self.change_lane > 0 and self.change_lane < self.CHANGE_LENGTH:
            self.change_lane += 1
            lane_im = self.add_change_sign(lane_im)
            return lane_im

        if self.NIGHT_MODE:
            lane_im = self._handle_dark_frame(lane_im)

        # Convert to RGB and grayscale
        frame_gray = cv2.cvtColor(lane_im, cv2.COLOR_BGR2GRAY)

        # Blur the image
        blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)

        if self.CROSSWALK:
            # Detect crosswalk
           lane_im = self.detect_crosswalk(frame=orig_frame)
    
        # Detect edges
        mag_im = cv2.Canny(blur, self.CANNY_TH[0], self.CANNY_TH[1])

        # We focus only on the region of interest
        mask = np.zeros_like(mag_im)

        vertices = np.array(
            object=[
                [
                    self.ROI["bottom_left"],
                    self.ROI["bottom_right"],
                    self.ROI["top_right"],
                    self.ROI["top_left"],
                ]
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(mask, vertices, 255)

        masked_im = cv2.bitwise_and(mag_im, mask)
        # Apply Hough transform
        lines = cv2.HoughLinesP(
            masked_im,
            self.R_STEP,
            self.THETA_STEP,
            self.HOUGH_TH,
            minLineLength=self.MIN_LINE_LENGTH,
            maxLineGap=self.MAX_LINE_GAP,
        )
        lines = lines if lines is not None else []

        # Find the mean lines
        left, right = self.find_mean_lines(lines, self.M_TH[0], self.M_TH[1])

        if self._no_lanes_initially(left, right):
            return lane_im

        if left is None and right is None:
            # Detected no lanes
            # Update the number of frames without lanes in a row
            self.frames_wo_lanes += 1

            if self.frames_wo_lanes > self.FILL_TH:
                if self.CROSSWALK:
                    processed_frame = lane_im
                    return processed_frame
                else:
                    # Been too long without lanes, probably changed lanes
                    self.change_lane = 1
                    lane_im = self.add_change_sign(lane_im)
        else:
            # Detected some lanes
            self.frames_wo_lanes = 0

        # Complete missing lanes if needed and update the last lanes
        left, right = self.complete_update(left, right)
        
        processed_frame = self._detect_lanes(lane_im, left, right)
        
        if self.CROSSWALK:
            processed_frame = lane_im
        return processed_frame

    def _handle_dark_frame(self, dark_frame: np.ndarray, strength=0.8, beta=0.1) -> np.ndarray:
        # Increase the brightness
        return cv2.convertScaleAbs(dark_frame, alpha=strength, beta=beta)

    def _no_lanes_initially(self, left: np.ndarray, right: np.ndarray) -> bool:
        """
        Check if there were no lanes detected initially.
        """
        return (self.last_left is None or self.last_right is None) and (
            left is None or right is None
        )
    def _detect_lanes(
        self,
        lane_im: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        # Extend the lines
        left, right = self.extend_lines(lines=[left, right], y1=self.LINES_Y[0], y2=self.LINES_Y[1])
        
        # Draw the mean extend lines
        for line in [left, right]:
            x1, y1, x2, y2 = line
            p1 = (x1, y1)
            p2 = (x2, y2)
            cv2.line(lane_im, p1, p2, (0, 255, 0), thickness=self.THICKNESS)
        
        if self.CURVE:
            self._update_curve_slopes(left, right)
            self._detect_curve(lane_im, left, right)
            
        return lane_im

    def _detect_curve(self, lane_im: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        l_x0, l_y0, _, _ = left
        r_x0, r_y0, _, _ = right
        m_x0, m_y0 = (int((l_x0 + r_x0) / 2 ), int((l_y0 + r_y0) / 2))
        
        self.latest_curve_x.append(m_x0)
        m_x0 = int(np.mean(self.latest_curve_x[-60:]))
                    
        slopes = self.curve_slopes[-50:]
        first_slop = -np.mean(slopes)
        
        for i in range(self.num_of_windows):
            t = i / (self.num_of_windows - 1)
            slope = (2*(t**1.3) - 1) * first_slop 
            
            m_y1 = int(m_y0 - self.window_height)
            m_x1 = int(m_x0 + (self.window_height * slope))
            cv2.line(lane_im, (m_x0, m_y0), (m_x1, m_y1), (0, 0, 255), thickness=int(self.THICKNESS / 2) + 1)
            m_x0, m_y0 = m_x1, m_y1

    def _update_curve_slopes(self, left: np.ndarray, right: np.ndarray):
        # Update the slopes
        left = left if left is not None else self.last_left
        right = right if right is not None else self.last_right
        
        left_slope = ((left[3] - left[1]) / (left[2] - left[0])) 
        right_slope = ((right[3] - right[1]) / (right[2] - right[0]))
        
        self.left_slopes.append(left_slope)
        self.right_slopes.append(right_slope)
        slope = (self.left_slopes[-1] + self.right_slopes[-1]) / 2
        self.curve_slopes.append(np.sign(slope) * ((abs(slope - 0.7)) ** 1.7))

    def complete_update(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Update the lane detector with the latest lanes and complete the missing lanes.
        Input:
            left: the left lane
            right: the right lane
        Output:
            left, right: the updated left and right lanes
        """
        if left is None:
            left = self.last_left
        else:
            self.last_left = left

        if right is None:
            right = self.last_right
        else:
            self.last_right = right

        return left, right


    def find_mean_lines(
        self, lines: np.ndarray, m_low: float, m_high: float
    ) -> np.ndarray:
        """
        Find the mean line for the left and right lanes.
        Input:
            lines: a set of lines in the form of (x1, y1, x2, y2)
            m_low: the lower bound for the slope of the lines
            m_high: the upper bound for the slope of the lines
        Output:
            lines: a set of lines in the form of (x1, y1, x2, y2)
            Number of lines is based on the number of lanes detected.
        """
        # Separate the lines into left and right
        valid_lines = []
        slopes = []
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                m = 0
            else:
                m, _ = np.polyfit((x1, x2), (y1, y2), 1)

            # Take slopes that are within a certain range that can be a lane
            if abs(m) < m_high and abs(m) > m_low:
                if m > 0:
                    left_lines.append(line[0])
                else:
                    right_lines.append(line[0])

                valid_lines.append(line[0])
                slopes.append(m)

        # Find the mean line for left and right
        left = self._find_mean_lines(left_lines)
        right = self._find_mean_lines(right_lines)

        return left, right

    def _find_mean_lines(self, lines: np.ndarray) -> np.ndarray:
        """
        Find the mean line from a set of lines.
        Input:
            lines: a set of lines in the form of (x1, y1, x2, y2)
        Output:
            line: a line in the form of (x1, y1, x2, y2)
        """
        if len(lines) == 0:
            return None
        elif len(lines) == 1:
            return lines[0]
        else:
            return np.array(lines).mean(axis=0).astype(int)

    def extend_lines(self, lines: np.ndarray, y1: int, y2: int) -> np.ndarray:
        """
        Extend the lines to the given y1 and y2.

        Input:
            lines: a set of lines in the form of (x1, y1, x2, y2)
            y1: the y coordinate of the top of the line
            y2: the y coordinate of the bottom of the line

        Output:
            lines: a set of lines in the form of (x1, y1, x2, y2)
        """
        extended_lines = []

        for line in lines:
            if line is None:
                continue

            if line[0] == line[2]:
                x1 = x2 = int(line[0])
            else:
                m, b = np.polyfit((line[0], line[2]), (line[1], line[3]), 1)

                if m == 0:
                    # Handle the case where the line is horizontal (parallel to x-axis)
                    x1 = x2 = int(line[0])  # Use the x-coordinate from the line data
                else:
                    x1 = int((y1 - b) / m)
                    x2 = int((y2 - b) / m)

            extended_lines.append([x1, y1, x2, y2])

        return np.array(extended_lines)

    def add_change_sign(self, lane_im: np.ndarray) -> np.ndarray:
        """
        Add a sign to the image to indicate a change in lanes.
        Input:
            lane_im: an image of the lanes
        Output:
            lane_im: an image of the lanes with a sign
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (
            lane_im.shape[1] // 2 - 100,
            lane_im.shape[0] // 2 - 200,
        )
        fontScale = 3
        fontColor = (0, 0, 255)
        lineType = 2

        cv2.putText(
            lane_im,
            "Changing Lanes",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )
        return lane_im

    
    def detect_crosswalk(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect crosswalk in the frame and draw bounding box around it.
        Input:
            frame: a single frame of a video
        Output:
            frame: the same frame with detected crosswalk drawn on it
        """
        ROI = {
        "bottom_left": (0, 1070),
        "bottom_right": (1900, 1070),
        "top_left": (100, 680),
        "top_right": (1350, 680),
        }
        
        # Leave only the white color
        lower = np.array([200, 200, 200])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(frame, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert to gray
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Apply region of interest
        mask = np.zeros_like(gray)
        vertices = np.array(
            object=[
                [
                    ROI["bottom_left"],
                    ROI["bottom_right"],
                    ROI["top_right"],
                    ROI["top_left"],
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, vertices, 255)
        masked = cv2.bitwise_and(gray, mask)

        # Apply Canny edge detection
        edges = cv2.Canny(masked, 50, 150)

        # Extend horizontal edges
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Detect contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area and sizes
        min_area = 28000
        min_width = 1000
        contours = [cnt for cnt in contours if 
                    cv2.contourArea(cnt) > min_area and cv2.contourArea(cnt) < self.max_area
                    and cv2.boundingRect(cnt)[2] > min_width]
        
        if len(contours) == 0:
            self.max_area = self.max_area_orig
        else:
            self.max_area *= 1.1

        # Draw the contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            yy = int(y + h * 0.25)
            cv2.rectangle(frame, (x, yy), (x + w, y + h - 10), (0, 0, 255), 8)

        return frame