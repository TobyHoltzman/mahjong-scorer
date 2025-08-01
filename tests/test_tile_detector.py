#!/usr/bin/env python3
"""
Test script to verify tile detection
"""

import cv2
import numpy as np
import argparse
from mahjong_scorer import TileDetector


def test_tile_detector(show_images: bool = False):
    """
    Test tile detection functionality.
    
    Args:
        show_images: If True, display visualization windows
    """
    # Hardcoded path to the test image
    FILEPATH_TILE_TEST_IMAGE = "tests/resources/toby_test.png"

    # Initialize the tile detector
    print("Initializing TileDetector...")
    detector = TileDetector(show_images=show_images)

    # Load the test image
    print(f"Loading test image from {FILEPATH_TILE_TEST_IMAGE}...")
    source_image = cv2.imread(FILEPATH_TILE_TEST_IMAGE)
    if source_image is None:
        print("Error: Could not load test image.")
        return
    
    cv2.imshow("Source Image", source_image)
    
    # Get tiles
    tiles = detector.detect_tiles(source_image)
    if not tiles:
        print("No tiles detected.")
        return
    
    print(f"Detected {len(tiles)} tiles.")
    # Draw detected tiles on the image
    for tile in tiles:
        cv2.imshow("Detected Tile", tile)
        cv2.waitKey(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test tile detection')
    parser.add_argument('--debug', action='store_true',
                      help='Run and show debug images')
    args = parser.parse_args()
    
    test_tile_detector(show_images=args.debug)