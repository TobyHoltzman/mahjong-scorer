"""Constants for mahjong scorer."""

import cv2
import numpy as np

# Detection parameters
DETECTION_PARAMS = {
    'min_contour_area': 100,
    'max_contour_area': 50000,
    'min_contour_area_tile': 1000,
    'max_aspect_ratio': 10.0,
    'agglomerative_dist': 20,
    'line_dist_tolerance': 100,
    'line_angle_tolerance': 10,
}

# Image preprocessing parameters
GAUSSIAN_BLUR_KERNEL = (7, 11)
GAUSSIAN_BLUR_SMALL = (3, 3)
CANNY_THRESHOLD_LOW = 50
CANNY_THRESHOLD_HIGH = 150
DILATION_KERNEL_SIZE = (3, 3)
DILATION_ITERATIONS = 1

SATURATION_CANNY_THRESHOLD_LOW_RATIO = 3
VALUE_CANNY_THRESHOLD_LOW_RATIO = 5

# Line detection parameters
HOUGH_RHO = 5  
HOUGH_THETA = np.pi / 180  
HOUGH_THRESHOLD = 50  
HOUGH_MIN_LINE_LENGTH_RATIO = 1/3  
HOUGH_MAX_LINE_GAP_RATIO = 1/12 

# Angle and Rotation parameters
HORIZONTAL_ANGLE_THRESHOLD = 5
VERTICAL_ANGLE_MIN = 60
VERTICAL_ANGLE_MAX = 120
HORIZONTAL_ANGLE_HIGH = 175

ANGLE_THRESHOLD_NEG = -45
ANGLE_THRESHOLD_POS = 45
ANGLE_ADJUSTMENT = 90

# Morphology parameters
MORPHOLOGY_KERNEL = np.ones((3, 3), np.uint8)
VERTICAL_LINE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))

# Visualization
DEBUG_MAX_WIDTH = 1400
DEBUG_MAX_HEIGHT = 800

# Default paths
DEFAULT_TEMPLATE_DIR = "templates"