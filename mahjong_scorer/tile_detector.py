"""
Tile detection module for mahjong scorer using OpenCV.
"""

from typing import List, Optional, Tuple
import cv2
import math
import numpy as np

from .constants import (
    CANNY_THRESHOLD_HIGH,
    CANNY_THRESHOLD_LOW,
    DEFAULT_TEMPLATE_DIR,
    DETECTION_PARAMS,
    DILATION_ITERATIONS,
    DILATION_KERNEL_SIZE,
    GAUSSIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_SMALL,
    HORIZONTAL_ANGLE_HIGH,
    HORIZONTAL_ANGLE_THRESHOLD,
    HOUGH_MAX_LINE_GAP_RATIO,
    HOUGH_MIN_LINE_LENGTH_RATIO,
    HOUGH_RHO,
    HOUGH_THETA,
    HOUGH_THRESHOLD,
    MORPHOLOGY_KERNEL,
    SATURATION_CANNY_THRESHOLD_LOW_RATIO,
    VALUE_CANNY_THRESHOLD_LOW_RATIO,
    VERTICAL_ANGLE_MAX,
    VERTICAL_ANGLE_MIN,
    VERTICAL_LINE_KERNEL,
)
from .tile_recognition import TileRecognizer
from .utils.image_utils import (
    extend_vertical_lines_mask,
    rotate_and_crop_cluster,
    crop_image_from_contour
)


class TileDetector:
    """Class for detecting mahjong tiles using computer vision."""

    def __init__(self, show_images: bool = False) -> None:
        """
        Initialize the tile detector.

        Args:
            show_images: If True, display visualization windows
        """
        self.recognizer = TileRecognizer()
        self.detection_params = DETECTION_PARAMS.copy()
        self.show_images = show_images
        self._debug_images = {}

    def show_debug_image(self, title: str, image: np.ndarray) -> None:
        """
        Store debug image for later display if visualization is enabled.

        Args:
            title: Window title
            image: Image to store
        """
        if self.show_images:
            self._debug_images[title] = image.copy()

    def display_debug_images(self) -> None:
        """Display all collected debug images if visualization is enabled."""
        if not self.show_images or not self._debug_images:
            return

        for title, image in self._debug_images.items():
            cv2.imshow(title, image)
        cv2.waitKey(0)
        self._debug_images.clear()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better tile detection.

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        edges = cv2.Canny(blurred, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        kernel = np.ones(DILATION_KERNEL_SIZE, np.uint8)
        return cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)

    def find_tile_clusters(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find clusters of tiles in the preprocessed image.

        Args:
            image: Preprocessed image

        Returns:
            List of contours representing potential tile clusters
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return [
            contour for contour in contours
            if cv2.contourArea(contour) > self.detection_params['min_contour_area']
        ]

    def filter_horizontal_vertical_lines(self, lines: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter lines to keep only horizontal and vertical ones.

        Args:
            lines: List of detected lines, each in format [[x1, y1, x2, y2]]

        Returns:
            List of filtered lines that are horizontal or vertical
        """
        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            # Consider lines within threshold of horizontal or vertical
            if (angle < HORIZONTAL_ANGLE_THRESHOLD or
                (VERTICAL_ANGLE_MIN < angle < VERTICAL_ANGLE_MAX) or
                angle > HORIZONTAL_ANGLE_HIGH):
                filtered_lines.append(line)

        return filtered_lines

    def find_tile_contours_from_cluster(self, source_image: np.ndarray, cluster: np.ndarray) -> List[np.ndarray]:
        """
        Find individual tile contours from a cluster.

        Args:
            source_image: Original image
            cluster: Contour representing a tile cluster

        Returns:
            List of individual tile contours
        """
        # Create a mask for the cluster
        mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cluster], -1, 255, thickness=cv2.FILLED)

        rotated_image = rotate_and_crop_cluster(source_image, cluster)
        rotated_mask = rotate_and_crop_cluster(mask, cluster)
        rotated_mask = rotate_and_crop_cluster(mask, cluster)

        self.show_debug_image("Rotated mask", rotated_mask)
        self.show_debug_image("Masked Image", rotated_image)

        # Convert to HSV color space and show each individual image
        hsv_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2HSV)
        self.show_debug_image("Hue", hsv_image[:, :, 0])
        self.show_debug_image("Saturation", hsv_image[:, :, 1])
        self.show_debug_image("Value", hsv_image[:, :, 2])

        # Threshold the saturation and value channel to create a binary mask
        _, value_mask = cv2.threshold(hsv_image[:, :, 2], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, saturation_mask = cv2.threshold(hsv_image[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        combined_mask = cv2.bitwise_or(value_mask, saturation_mask)
        closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, MORPHOLOGY_KERNEL)
        self.show_debug_image("Closed Mask", closed_mask)

        # Find contours in the closed mask to isolate tile borders
        contours, _ = cv2.findContours(cv2.bitwise_not(closed_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border_contour = max(contours, key=cv2.contourArea)
        border_mask = np.zeros_like(closed_mask)
        cv2.drawContours(border_mask, [border_contour], -1, 255, thickness=cv2.FILLED)
        closed_mask = cv2.bitwise_and(closed_mask, border_mask)
        dilated_mask = cv2.dilate(closed_mask, MORPHOLOGY_KERNEL, iterations=1)
        self.show_debug_image("Tile text mask", dilated_mask)

        # Apply mask to the cropped image to remove text and other artifacts
        masked_cropped_image = cv2.bitwise_and(rotated_image, rotated_image, mask=cv2.bitwise_not(dilated_mask))
        self.show_debug_image("Masked Cropped Image", masked_cropped_image)

        # Apply Gaussian blur using on sauration and value channels
        saturation_blurred = cv2.GaussianBlur(hsv_image[:, :, 1], GAUSSIAN_BLUR_SMALL, 0)
        value_blurred = cv2.GaussianBlur(hsv_image[:, :, 2], GAUSSIAN_BLUR_SMALL, 0)

        # Find Canny threshold values from Otsu's method
        saturation_otsu_threshold, _  = cv2.threshold(saturation_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        saturation_canny_threshold_low = saturation_otsu_threshold / SATURATION_CANNY_THRESHOLD_LOW_RATIO
        saturation_canny_threshold_high = saturation_otsu_threshold

        value_otsu_threshold, _ = cv2.threshold(value_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        value_canny_threshold_low = value_otsu_threshold / VALUE_CANNY_THRESHOLD_LOW_RATIO
        value_canny_threshold_high = value_otsu_threshold

        # Apply Canny edge detection to the Saturation and Value channels
        saturated_edges = cv2.Canny(saturation_blurred, saturation_canny_threshold_low, saturation_canny_threshold_high)
        self.show_debug_image("Saturated Edges", saturated_edges)

        value_edges = cv2.Canny(value_blurred, 0, value_canny_threshold_low, value_canny_threshold_high)
        self.show_debug_image("Value Edges", value_edges)

        # OR the edges with the saturated and value edges
        edges = cv2.bitwise_or(value_edges, saturated_edges)
        self.show_debug_image("Combined Edges", edges)

        # Subtract the dilated mask from the edges
        edges = cv2.bitwise_and(edges, cv2.bitwise_not(dilated_mask))
        self.show_debug_image("Edges after Masking", edges)

        # Close the edges several times to fill gaps
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, VERTICAL_LINE_KERNEL, iterations=1)
        self.show_debug_image("Closed Edges", closed_edges)

        # Find longest line segments in the edges
        h = masked_cropped_image.shape[0]
        lines = cv2.HoughLinesP(
            closed_edges,
            HOUGH_RHO,
            HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=h * HOUGH_MIN_LINE_LENGTH_RATIO,
            maxLineGap=h * HOUGH_MAX_LINE_GAP_RATIO
        )

        # Filter for only vertical and horizontal lines
        filtered_lines = self.filter_horizontal_vertical_lines(lines)

        # Extend the lines to cover the entire height of the image
        line_mask = extend_vertical_lines_mask(filtered_lines, masked_cropped_image.shape)
        self.show_debug_image("Extended Lines Mask", line_mask)

        # Find contours in the line mask
        line_contours, _ = cv2.findContours(cv2.bitwise_not(line_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the masked cropped image and crop individual tiles
        tiles = []
        contour_image = masked_cropped_image.copy()
        for contour in line_contours:
            # Filter contours base on area
            area = cv2.contourArea(contour)
            if self.detection_params['min_contour_area_tile'] < area < self.detection_params['max_contour_area']:
                cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
                tile = crop_image_from_contour(rotated_image, contour)
                tiles.append(tile)
                
        self.show_debug_image("Detected Lines", contour_image)

        self.display_debug_images()

        return tiles
    
    def detect_tiles(self, source_image: np.ndarray) -> List[np.ndarray]:
        """
        Detect and recognize tiles in the source image.

        Args:
            source_image: Input image (BGR format)
        Returns:
            List of detected tiles
        """

        # Preprocess the image
        processed = self.preprocess_image(source_image)

        # Find tile clusters
        tile_clusters = self.find_tile_clusters(processed)
        self.show_debug_image("Tile Clusters", processed)

        detected_tiles = []
        for cluster in tile_clusters:
            tiles = self.find_tile_contours_from_cluster(source_image, cluster)

            if not tiles:
                print("No tiles found in cluster.")
                continue
                    
            detected_tiles.extend(tiles)

        return detected_tiles