"""Image processing utility functions for mahjong scorer."""

import cv2
import numpy as np
from typing import List, Tuple
from ..constants import (
    ANGLE_THRESHOLD_NEG, ANGLE_THRESHOLD_POS, ANGLE_ADJUSTMENT,
    DETECTION_PARAMS
)

def calculate_contour_distance(contour1: np.ndarray, contour2: np.ndarray) -> float:
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    c1 = (x1 + w1 / 2, y1 + h1 / 2)
    c2 = (x2 + w2 / 2, y2 + h2 / 2)
    return max(abs(c1[0] - c2[0]) - (w1 + w2) / 2, abs(c1[1] - c2[1]) - (h1 + h2) / 2)

def agglomerative_cluster(contours: List[np.ndarray], threshold_distance: float) -> List[np.ndarray]:
    curr = list(contours)
    while len(curr) > 1:
        min_dist, pair = None, None
        for x in range(len(curr) - 1):
            for y in range(x + 1, len(curr)):
                d = calculate_contour_distance(curr[x], curr[y])
                if min_dist is None or d < min_dist:
                    min_dist, pair = d, (x, y)
        if pair and min_dist < threshold_distance:
            i, j = pair
            curr[i] = np.concatenate((curr[i], curr[j]), axis=0)
            del curr[j]
        else:
            break
    return curr

def filter_contours_by_aspect_ratio(contours: List[np.ndarray], min_area: float) -> List[np.ndarray]:
    filtered = []
    ratio_limit = DETECTION_PARAMS['max_aspect_ratio']
    for cnt in contours:
        (_, _), (w, h), _ = cv2.minAreaRect(cnt)
        if w == 0 or h == 0: continue
        if (max(w, h) / min(w, h)) < ratio_limit and cv2.contourArea(cnt) > min_area:
            filtered.append(cnt)
    return filtered

def is_line_duplicate(line: np.ndarray, existing: List[np.ndarray]) -> bool:
    l1 = line[0]
    ang1 = np.degrees(np.arctan2(l1[3] - l1[1], l1[2] - l1[0])) % 180
    mid1 = np.array([(l1[0] + l1[2]) / 2, (l1[1] + l1[3]) / 2])
    for ex in existing:
        l2 = ex[0]
        ang2 = np.degrees(np.arctan2(l2[3] - l2[1], l2[2] - l2[0])) % 180
        mid2 = np.array([(l2[0] + l2[2]) / 2, (l2[1] + l2[3]) / 2])
        if abs(ang1 - ang2) <= DETECTION_PARAMS['line_angle_tolerance'] and \
           np.linalg.norm(mid1 - mid2) < DETECTION_PARAMS['line_dist_tolerance']:
            return True
    return False

def rotate_and_crop_cluster(image: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(cluster)
    (cx, cy), (w, h), angle = rect
    if angle < ANGLE_THRESHOLD_NEG:
        angle += ANGLE_ADJUSTMENT
        w, h = h, w
    elif angle > ANGLE_THRESHOLD_POS:
        angle -= ANGLE_ADJUSTMENT
        w, h = h, w
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    ih, iw = image.shape[:2]
    cos_a, sin_a = abs(np.cos(np.radians(angle))), abs(np.sin(np.radians(angle)))
    nw, nh = int(ih * sin_a + iw * cos_a), int(ih * cos_a + iw * sin_a)
    M[0, 2] += (nw / 2) - cx
    M[1, 2] += (nh / 2) - cy
    
    rotated = cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR)
    box = np.int32(cv2.boxPoints(rect))
    pts = cv2.transform(np.array([box]), M)[0]
    x, y = max(0, int(np.min(pts[:, 0]))), max(0, int(np.min(pts[:, 1])))
    x2, y2 = min(nw, int(np.max(pts[:, 0]))), min(nh, int(np.max(pts[:, 1])))
    return rotated[y:y2, x:x2]

def extend_vertical_lines_mask(lines: List[np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        dx, dy = x2 - x1, y2 - y1
        if dx == 0: p1, p2 = (int(x1), 0), (int(x1), h - 1)
        else:
            m, b = dy / dx, y1 - (dy / dx) * x1
            pts = []
            for xv in [0, w-1]:
                yv = int(m * xv + b)
                if 0 <= yv < h: pts.append((xv, yv))
            for yv in [0, h-1]:
                xv = int((yv - b) / m)
                if 0 <= xv < w: pts.append((xv, yv))
            p1, p2 = (pts[0], pts[1]) if len(pts) >= 2 else ((x1, y1), (x2, y2))
        cv2.line(mask, p1, p2, 255, 2)
    return mask

def crop_image_from_contour(image: np.ndarray, contour: np.ndarray) -> np.ndarray:
    x, y, w, h = cv2.boundingRect(contour)
    return image[y:y+h, x:x+w] if w > 0 and h > 0 else np.zeros((1, 1, 3), dtype=np.uint8)