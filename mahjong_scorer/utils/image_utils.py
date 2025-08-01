"""Image processing utility functions for mahjong scorer."""

import cv2
import math
import numpy as np
from typing import List, Tuple

# Angle thresholds for rotation
ANGLE_THRESHOLD_NEG = -45
ANGLE_THRESHOLD_POS = 45
ANGLE_ADJUSTMENT = 90

# Line detection parameters
VERTICAL_ANGLE_MIN = 80
VERTICAL_ANGLE_MAX = 100
LINE_THICKNESS = 2
LINE_COLOR = 255

# Image type constants
UINT8_MAX = 255
BORDER_VALUE = (0, 0, 0)

def rotate_and_crop_cluster(source_image: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    """
    Rotate and crop image based on cluster contour.
    
    Args:
        source_image: Original image
        cluster: Contour representing a tile cluster
        
    Returns:
        Rotated and cropped image
    """
    # Find minimum area rectangle and convert to box points
    min_area_rect = cv2.minAreaRect(cluster)
    box = cv2.boxPoints(min_area_rect)
    box = np.int32(box)
    
    # Get rectangle properties
    (center_x, center_y), (width, height), angle = min_area_rect
    
    # Adjust angle for proper orientation
    if angle < ANGLE_THRESHOLD_NEG:
        angle += ANGLE_ADJUSTMENT
        width, height = height, width
    elif angle > ANGLE_THRESHOLD_POS:
        angle -= ANGLE_ADJUSTMENT
        width, height = height, width
    
    # Create rotation matrix around center
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # Calculate new dimensions after rotation
    img_height, img_width = source_image.shape[:2]
    cos_angle = abs(np.cos(np.radians(angle)))
    sin_angle = abs(np.sin(np.radians(angle)))
    new_width = int(img_height * sin_angle + img_width * cos_angle)
    new_height = int(img_height * cos_angle + img_width * sin_angle)
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center_x
    rotation_matrix[1, 2] += (new_height / 2) - center_y
    
    # Apply rotation to image
    rotated_image = cv2.warpAffine(
        source_image, 
        rotation_matrix, 
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_VALUE
    )
    
    # Calculate new bounding box coordinates
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]
    min_x = max(0, int(np.min(rotated_box[:, 0])))
    max_x = min(new_width, int(np.max(rotated_box[:, 0])))
    min_y = max(0, int(np.min(rotated_box[:, 1])))
    max_y = min(new_height, int(np.max(rotated_box[:, 1])))
    
    # Return cropped rotated image
    return rotated_image[min_y:max_y, min_x:max_x]

def extend_vertical_lines_mask(lines: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask of extended vertical lines.
    
    Args:
        lines: List of lines [x1, y1, x2, y2]
        image_shape: Tuple of (height, width) for output mask size
        
    Returns:
        Binary mask with extended vertical lines
    """
    # Create empty mask with image dimensions
    h, w = image_shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Process each line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        # Handle vertical lines (within threshold of vertical)
        if VERTICAL_ANGLE_MIN < angle < VERTICAL_ANGLE_MAX:
            # Calculate midpoint x-coordinate for vertical line
            x_mid = int((x1 + x2) / 2)
            # Draw extended vertical line on mask
            cv2.line(line_mask, (x_mid, 0), (x_mid, h-1), LINE_COLOR, LINE_THICKNESS)
        else:
            # Draw original line for non-vertical lines
            cv2.line(line_mask, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)
    
    return line_mask

def crop_image_from_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Crop a region from the image based on the provided contour.
    
    Args:
        image: Source image
        contour: Contour to crop
        
    Returns:
        Cropped image region
    """
    # Get bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Ensure valid dimensions
    if w <= 0 or h <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    
    # Crop and return the region
    return image[y:y+h, x:x+w]