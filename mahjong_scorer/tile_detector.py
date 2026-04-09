"""Tile detection module for mahjong scorer using OpenCV."""

from typing import List, Optional
import cv2
import math
import numpy as np

from .constants import (
    CANNY_THRESHOLD_HIGH, CANNY_THRESHOLD_LOW, DETECTION_PARAMS,
    DILATION_ITERATIONS, GAUSSIAN_BLUR_KERNEL, GAUSSIAN_BLUR_SMALL, 
    HOUGH_MAX_LINE_GAP_RATIO, HOUGH_MIN_LINE_LENGTH_RATIO, HOUGH_RHO, 
    HOUGH_THETA, HOUGH_THRESHOLD, MORPHOLOGY_KERNEL,
    SATURATION_CANNY_THRESHOLD_LOW_RATIO, VALUE_CANNY_THRESHOLD_LOW_RATIO,
    VERTICAL_ANGLE_MAX, VERTICAL_ANGLE_MIN, VERTICAL_LINE_KERNEL,
    DEBUG_MAX_WIDTH, DEBUG_MAX_HEIGHT
)
from .tile_recognition import TileRecognizer
from .utils import image_utils as utils

class TileDetector:
    def __init__(self, show_images: bool = False) -> None:
        self.recognizer = TileRecognizer()
        self.params = DETECTION_PARAMS
        self.show_images = show_images
        self._debug_images = {}

    def show_debug_image(self, title: str, image: np.ndarray) -> None:
        if self.show_images: self._debug_images[title] = image.copy()

    def display_debug_images(self) -> None:
        if not self.show_images or not self._debug_images: return
        for title, img in self._debug_images.items():
            h, w = img.shape[:2]
            scale = min(DEBUG_MAX_WIDTH / w, DEBUG_MAX_HEIGHT / h, 1.0)
            cv2.imshow(title, cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self._debug_images.clear()

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_KERNEL, 0)
        edges = cv2.Canny(blur, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        return cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=DILATION_ITERATIONS)

    def find_tile_clusters(self, image: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > self.params['min_contour_area']]

    def get_vertical_lines(self, lines: Optional[np.ndarray]) -> List[np.ndarray]:
        if lines is None: return []
        res = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if VERTICAL_ANGLE_MIN < angle < VERTICAL_ANGLE_MAX:
                res.append(line)
        return res

    def find_tile_contours_from_cluster(self, source_image: np.ndarray, cluster: np.ndarray) -> List[np.ndarray]:
        # Isolate and straighten
        mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cluster], -1, 255, thickness=cv2.FILLED)
        rot_img = utils.rotate_and_crop_cluster(cv2.bitwise_and(source_image, source_image, mask=mask), cluster)
        rot_mask = utils.rotate_and_crop_cluster(mask, cluster)
        hsv = cv2.cvtColor(rot_img, cv2.COLOR_BGR2HSV)

        # Content masking
        s_blur = cv2.GaussianBlur(hsv[:, :, 1], GAUSSIAN_BLUR_SMALL, 0)
        
        # otsu_s can be an array depending on OpenCV version/input
        _, otsu_s_raw = cv2.threshold(s_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert to float safely regardless of whether it's a scalar or array
        otsu_s = float(np.max(otsu_s_raw)) 
        
        s_low = otsu_s / SATURATION_CANNY_THRESHOLD_LOW_RATIO
        s_high = otsu_s
        
        s_edges = cv2.Canny(s_blur, s_low, s_high)
        s_edges = cv2.bitwise_and(s_edges, cv2.erode(rot_mask, MORPHOLOGY_KERNEL, iterations=3))
        
        cnts, _ = cv2.findContours(cv2.morphologyEx(s_edges, cv2.MORPH_CLOSE, MORPHOLOGY_KERNEL, iterations=3), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return []

        valid_cnts = utils.filter_contours_by_aspect_ratio(cnts, self.params['min_contour_area_tile'])
        clustered = utils.agglomerative_cluster(valid_cnts, self.params['agglomerative_dist'])

        t_mask = np.zeros(rot_img.shape[:2], dtype=np.uint8)
        for c in clustered: cv2.fillPoly(t_mask, [cv2.convexHull(c).astype(int)], 255)
        dilated_t_mask = cv2.dilate(t_mask, MORPHOLOGY_KERNEL, iterations=5)

        # Vertical separator detection
        v_blur = cv2.GaussianBlur(hsv[:, :, 2], GAUSSIAN_BLUR_SMALL, 0)
        _, otsu_v_raw = cv2.threshold(v_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        otsu_v = float(np.max(otsu_v_raw))
        
        v_low = otsu_v / VALUE_CANNY_THRESHOLD_LOW_RATIO
        v_high = otsu_v
        
        v_edges = cv2.Canny(v_blur, v_low, v_high)
        
        edges = cv2.bitwise_and(cv2.bitwise_or(v_edges, s_edges), cv2.bitwise_not(dilated_t_mask))
        cleaned = cv2.Canny(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, VERTICAL_LINE_KERNEL), CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

        h, _ = rot_img.shape[:2]
        lines_p = cv2.HoughLinesP(cleaned, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, 
                                 minLineLength=int(h * HOUGH_MIN_LINE_LENGTH_RATIO), 
                                 maxLineGap=int(h * HOUGH_MAX_LINE_GAP_RATIO))
        
        final_lines = []
        for line in sorted(self.get_vertical_lines(lines_p), key=lambda l: np.linalg.norm(l[0][:2] - l[0][2:]), reverse=True):
            if not utils.is_line_duplicate(line, final_lines):
                final_lines.append(line)

        # Extraction
        l_mask = utils.extend_vertical_lines_mask(final_lines, rot_img.shape)
        tile_cnts, _ = cv2.findContours(cv2.bitwise_not(l_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        res_tiles = [utils.crop_image_from_contour(rot_img, c) for c in tile_cnts 
                     if self.params['min_contour_area_tile'] < cv2.contourArea(c) < self.params['max_contour_area']]

        self.display_debug_images()
        return res_tiles

    def detect_tiles(self, source_image: np.ndarray) -> List[np.ndarray]:
        clusters = self.find_tile_clusters(self.preprocess_image(source_image))
        all_tiles = []
        for cluster in clusters:
            tiles = self.find_tile_contours_from_cluster(source_image, cluster)
            if tiles: all_tiles.extend(tiles)
        return all_tiles