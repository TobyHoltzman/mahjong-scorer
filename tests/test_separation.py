#!/usr/bin/env python3
"""
Test script to verify separation of tile detection and recognition.
"""

import cv2
import numpy as np
from mahjong_scorer import TileDetector, TileRecognizer


def test_separation():
    """Test that detection and recognition are properly separated."""
    print("Testing Tile Detection and Recognition Separation")
    print("=" * 50)
    
    # Test 1: Independent recognizer
    print("\n1. Testing independent TileRecognizer:")
    recognizer = TileRecognizer()
    
    # Load templates
    success = recognizer.load_templates()
    print(f"Templates loaded: {success}")
    print(f"Available tiles: {len(recognizer.get_available_tiles())}")
    
    # Test recognition with a template
    if success:
        template_path = "templates/1m.png"
        template_img = cv2.imread(template_path)
        if template_img is not None:
            recognized, confidence = recognizer.recognize_tile_with_confidence(template_img)
            print(f"Recognized 1m template as: {recognized} (confidence: {confidence:.3f})")
    
    # Test 2: Independent detector
    print("\n2. Testing independent TileDetector:")
    detector = TileDetector()
    
    # Create a simple test image
    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)  # Dark background
    
    # Add a rectangle to simulate a tile
    cv2.rectangle(test_img, (100, 50), (180, 170), (255, 255, 255), 2)
    
    # Test detection without recognition
    processed = detector.preprocess_image(test_img)
    contours = detector.find_tile_contours(processed)
    print(f"Detected {len(contours)} potential tiles")
    
    # Test 3: Combined detection and recognition
    print("\n3. Testing combined detection and recognition:")
    
    # Load templates into detector
    detector.load_tile_templates()
    
    # Detect and recognize tiles
    detected_tiles = detector.detect_tiles(test_img)
    print(f"Detected and recognized {len(detected_tiles)} tiles")
    
    # Test 4: Parameter control
    print("\n4. Testing parameter control:")
    
    # Check current parameters
    params = detector.get_detection_parameters()
    print(f"Current detection parameters: {params}")
    
    # Modify parameters
    detector.set_detection_parameters(min_contour_area=500)
    detector.set_confidence_threshold(0.6)
    
    new_params = detector.get_detection_parameters()
    print(f"Modified detection parameters: {new_params}")
    
    # Test 5: Available tiles
    print("\n5. Testing available tiles:")
    available_tiles = detector.get_available_tiles()
    print(f"Available tiles: {len(available_tiles)}")
    print(f"Sample tiles: {available_tiles[:5]}")
    
    print("\n" + "=" * 50)
    print("Separation test completed!")
    print("\nBenefits of separation:")
    print("✓ Detection and recognition are independent")
    print("✓ Can use different recognition methods")
    print("✓ Easier to test and debug")
    print("✓ More modular and maintainable")


def test_recognizer_only():
    """Test using only the recognizer for known tile images."""
    print("\n\nTesting Recognizer-Only Usage")
    print("=" * 35)
    
    recognizer = TileRecognizer()
    recognizer.load_templates()
    
    # Test with multiple templates
    test_tiles = ['1m', '5p', '9s', 'east', 'red']
    
    for tile_name in test_tiles:
        template_path = f"templates/{tile_name}.png"
        template_img = cv2.imread(template_path)
        
        if template_img is not None:
            recognized, confidence = recognizer.recognize_tile_with_confidence(template_img)
            print(f"{tile_name}: {recognized} (confidence: {confidence:.3f})")
    
    print("\nRecognizer-only test completed!")


if __name__ == "__main__":
    test_separation()
    test_recognizer_only() 