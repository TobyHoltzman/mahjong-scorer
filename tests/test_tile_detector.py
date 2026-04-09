#!/usr/bin/env python3
"""
Test script to verify tile detection
"""

import cv2
import numpy as np
import argparse
from mahjong_scorer import TileDetector


def scale_image_for_display(image: np.ndarray, max_width: int = 1400, max_height: int = 800) -> np.ndarray:
    """
    Scale image to fit within monitor resolution limits.
    
    Args:
        image: Input image
        max_width: Maximum width in pixels
        max_height: Maximum height in pixels
        
    Returns:
        Scaled image
    """
    h, w = image.shape[:2]
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)

    if scale < 1.0:
        return cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def test_tile_detector(show_images: bool = False):
    """
    Test tile detection functionality.
    
    Args:
        show_images: If True, display visualization windows
    """
    # Hardcoded path to the test image
    FILEPATH_TILE_TEST_IMAGE = "tests/resources/toby_test.jpeg"

    # Initialize the tile detector
    print("Initializing TileDetector...")
    detector = TileDetector(show_images=show_images)

    # Load the test image
    print(f"Loading test image from {FILEPATH_TILE_TEST_IMAGE}...")
    source_image = cv2.imread(FILEPATH_TILE_TEST_IMAGE)
    if source_image is None:
        print("Error: Could not load test image.")
        return
    
    scaled_image = scale_image_for_display(source_image)
    cv2.imshow("Source Image", scaled_image)
    cv2.waitKey(1)
    
    # Get tiles
    tiles = detector.detect_tiles(source_image)
    if not tiles:
        print("No tiles detected.")
        return
    
    print(f"Detected {len(tiles)} tiles.")
    # Draw detected tiles on the image
    for i, tile in enumerate(tiles):
        scaled_tile = scale_image_for_display(tile)
        cv2.imshow(f"Detected Tile {i+1}", scaled_tile)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test tile detection')
    parser.add_argument('--debug', action='store_true',
                      help='Run and show debug images')
    args = parser.parse_args()
    
    test_tile_detector(show_images=args.debug)