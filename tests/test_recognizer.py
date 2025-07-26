#!/usr/bin/env python3
"""
Simple script to test mahjong tile recognition on an arbitrary image.
Usage: python test_recognizer.py <image_path> [model_path] [confidence_threshold]
"""

import sys
import cv2
import os
from mahjong_scorer.tile_recognition import TileRecognizer

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_recognizer.py <image_path> [model_path] [confidence_threshold]")
        print("Example: python test_recognizer.py test_tile.jpg models/mahjong_cnn.pth 0.7")
        sys.exit(1)
    
    # Parse arguments
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/mahjong_cnn.pth"
    confidence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        sys.exit(1)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        sys.exit(1)
    
    # Initialize recognizer
    recognizer = TileRecognizer(min_confidence=confidence_threshold, model_path=model_path)
    
    # Check if model loaded
    if recognizer.model is None:
        print(f"Error: Could not load model from '{model_path}'")
        sys.exit(1)
    
    # Recognize tile
    tile_name, confidence = recognizer.recognize_tile_with_confidence(image)
    
    # Print results
    print(f"\nImage: {image_path}")
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"Predicted tile: {tile_name or 'Unknown'}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Result: {'✓ Confident' if confidence >= confidence_threshold else '✗ Below threshold'}")
    
    # Show available tiles
    available_tiles = recognizer.get_available_tiles()
    print(f"\nAvailable tile classes ({len(available_tiles)}):")
    print(", ".join(available_tiles))

if __name__ == "__main__":
    main() 