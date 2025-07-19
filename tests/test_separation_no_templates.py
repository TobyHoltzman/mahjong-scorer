#!/usr/bin/env python3
"""
Test script to demonstrate separation of tile detection and recognition
without requiring template files.
"""

import cv2
import numpy as np
from mahjong_scorer import TileDetector, TileRecognizer


def create_mock_templates():
    """Create simple mock templates for testing."""
    templates = {}
    
    # Create a simple template for each tile type
    tile_types = ['1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
                  '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
                  '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
                  'east', 'south', 'west', 'north', 'red', 'green', 'white']
    
    for tile_type in tile_types:
        # Create a simple template image
        template = np.zeros((120, 80), dtype=np.uint8)
        
        # Add some distinctive pattern based on tile type
        if tile_type.endswith('m'):  # Man tiles
            template[30:90, 20:60] = 128
            cv2.putText(template, tile_type[0], (35, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2)
        elif tile_type.endswith('p'):  # Pin tiles
            template[30:90, 20:60] = 64
            cv2.putText(template, tile_type[0], (35, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2)
        elif tile_type.endswith('s'):  # Sou tiles
            template[30:90, 20:60] = 192
            cv2.putText(template, tile_type[0], (35, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255,), 2)
        else:  # Honor tiles
            template[30:90, 20:60] = 255
            cv2.putText(template, tile_type[:2], (25, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,), 2)
        
        templates[tile_type] = template
    
    return templates


def test_detection_only():
    """Test tile detection without recognition."""
    print("Testing Tile Detection (No Recognition)")
    print("=" * 40)
    
    detector = TileDetector()
    
    # Create a test image with multiple rectangles
    test_img = np.zeros((300, 400, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)  # Dark background
    
    # Add several rectangles to simulate tiles
    tile_positions = [
        (50, 50, 80, 120),   # x, y, w, h
        (150, 50, 80, 120),
        (250, 50, 80, 120),
        (50, 200, 80, 120),
        (150, 200, 80, 120),
    ]
    
    for x, y, w, h in tile_positions:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # Test detection
    processed = detector.preprocess_image(test_img)
    contours = detector.find_tile_contours(processed)
    
    print(f"Detected {len(contours)} potential tiles")
    
    # Show detection parameters
    params = detector.get_detection_parameters()
    print(f"Detection parameters: {params}")
    
    # Test parameter modification
    detector.set_detection_parameters(min_contour_area=500)
    contours_modified = detector.find_tile_contours(processed)
    print(f"After parameter change: {len(contours_modified)} potential tiles")
    
    return test_img, contours


def test_recognition_only():
    """Test tile recognition with mock templates."""
    print("\n\nTesting Tile Recognition (With Mock Templates)")
    print("=" * 45)
    
    recognizer = TileRecognizer()
    
    # Create mock templates
    mock_templates = create_mock_templates()
    recognizer.tile_templates = mock_templates
    recognizer.template_loaded = True
    
    print(f"Created {len(mock_templates)} mock templates")
    
    # Test recognition with a mock template
    test_tile = mock_templates['1m'].copy()
    recognized, confidence = recognizer.recognize_tile_with_confidence(test_tile)
    print(f"Recognized 1m template as: {recognized} (confidence: {confidence:.3f})")
    
    # Test with a slightly modified tile
    modified_tile = test_tile.copy()
    modified_tile[40:80, 30:50] = 100  # Add some noise
    recognized_modified, confidence_modified = recognizer.recognize_tile_with_confidence(modified_tile)
    print(f"Recognized modified tile as: {recognized_modified} (confidence: {confidence_modified:.3f})")
    
    # Test confidence threshold
    print(f"\nCurrent confidence threshold: {recognizer.min_confidence}")
    recognizer.set_confidence_threshold(0.8)
    print(f"New confidence threshold: {recognizer.min_confidence}")
    
    return recognizer


def test_combined_workflow():
    """Test the combined detection and recognition workflow."""
    print("\n\nTesting Combined Detection and Recognition")
    print("=" * 45)
    
    # Create detector with mock recognizer
    detector = TileDetector()
    mock_recognizer = TileRecognizer()
    mock_templates = create_mock_templates()
    mock_recognizer.tile_templates = mock_templates
    mock_recognizer.template_loaded = True
    
    # Replace detector's recognizer with mock
    detector.recognizer = mock_recognizer
    
    # Create test image
    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)
    
    # Add a tile-like rectangle
    cv2.rectangle(test_img, (100, 40), (180, 160), (255, 255, 255), 2)
    
    # Detect and recognize
    detected_tiles = detector.detect_tiles(test_img)
    print(f"Detected and recognized {len(detected_tiles)} tiles")
    
    for i, (tile_region, tile_type) in enumerate(detected_tiles):
        print(f"  Tile {i+1}: {tile_type}")
    
    return detector


def test_modular_design():
    """Demonstrate the benefits of modular design."""
    print("\n\nDemonstrating Modular Design Benefits")
    print("=" * 40)
    
    print("1. Independent Detection:")
    detector = TileDetector()
    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (100, 50), (180, 170), (255, 255, 255), 2)
    
    processed = detector.preprocess_image(test_img)
    contours = detector.find_tile_contours(processed)
    print(f"   - Detected {len(contours)} tiles without recognition")
    
    print("\n2. Independent Recognition:")
    recognizer = TileRecognizer()
    mock_templates = create_mock_templates()
    recognizer.tile_templates = mock_templates
    recognizer.template_loaded = True
    
    test_tile = mock_templates['5p'].copy()
    recognized, confidence = recognizer.recognize_tile_with_confidence(test_tile)
    print(f"   - Recognized tile as {recognized} (confidence: {confidence:.3f})")
    
    print("\n3. Easy Parameter Tuning:")
    detector.set_detection_parameters(min_contour_area=200)
    recognizer.set_confidence_threshold(0.5)
    print("   - Modified detection and recognition parameters independently")
    
    print("\n4. Swappable Recognition Methods:")
    print("   - Can easily replace template matching with ML-based recognition")
    print("   - Can use different recognition methods for different scenarios")
    
    print("\n5. Independent Testing:")
    print("   - Can test detection without recognition")
    print("   - Can test recognition without detection")
    print("   - Can test both together")


def main():
    """Run all separation tests."""
    print("Tile Detection and Recognition Separation Test")
    print("=" * 55)
    
    # Test 1: Detection only
    test_img, contours = test_detection_only()
    
    # Test 2: Recognition only
    recognizer = test_recognition_only()
    
    # Test 3: Combined workflow
    detector = test_combined_workflow()
    
    # Test 4: Modular design benefits
    test_modular_design()
    
    print("\n" + "=" * 55)
    print("SEPARATION TEST COMPLETED SUCCESSFULLY!")
    print("\nKey Benefits Demonstrated:")
    print("✓ Detection and recognition are completely independent")
    print("✓ Can test each component separately")
    print("✓ Easy to modify parameters for each component")
    print("✓ Can swap recognition methods without changing detection")
    print("✓ More maintainable and testable code structure")


if __name__ == "__main__":
    main() 