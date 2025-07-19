#!/usr/bin/env python3
"""
Example usage of the Mahjong Scorer with OpenCV.
"""

import cv2
import numpy as np
from mahjong_scorer import TileDetector, MahjongScorer


def main():
    """Demonstrate the mahjong scorer functionality."""
    print("Mahjong Scorer Example")
    print("=" * 30)
    
    # Initialize the scorer
    scorer = MahjongScorer()
    
    # Example 1: Manual tile input
    print("\n1. Manual tile analysis:")
    example_tiles = ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', 'east', 'east', 'east', 'red']
    
    print(f"Hand: {example_tiles}")
    
    # Check if valid
    is_valid = scorer.is_valid_hand(example_tiles)
    print(f"Valid hand: {is_valid}")
    
    # Count yaku
    yaku_list = scorer.count_yaku(example_tiles)
    print(f"Yaku found: {yaku_list}")
    
    # Calculate score
    total_han = sum(han for _, han in yaku_list)
    score_info = scorer.calculate_score(example_tiles, total_han)
    print(f"Score: {score_info}")
    
    # Example 2: Create a test image
    print("\n2. Image analysis example:")
    
    # Create a simple test image (simulating mahjong tiles)
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Dark gray background
    
    # Draw some rectangles to simulate tiles
    tile_positions = [
        (50, 100, 80, 120),   # x, y, w, h
        (150, 100, 80, 120),
        (250, 100, 80, 120),
        (350, 100, 80, 120),
        (50, 250, 80, 120),
        (150, 250, 80, 120),
        (250, 250, 80, 120),
        (350, 250, 80, 120),
        (450, 100, 80, 120),
        (450, 250, 80, 120),
        (550, 100, 80, 120),
        (550, 250, 80, 120),
        (50, 400, 80, 120),
    ]
    
    for i, (x, y, w, h) in enumerate(tile_positions):
        # Draw tile rectangle
        cv2.rectangle(test_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Add tile number
        cv2.putText(test_image, str(i + 1), (x + 30, y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save the test image
    cv2.imwrite('test_mahjong_hand.jpg', test_image)
    print("Created test image: test_mahjong_hand.jpg")
    
    # Analyze the image (this will detect contours but not recognize specific tiles yet)
    print("\nAnalyzing test image...")
    analysis = scorer.analyze_hand_from_image(test_image)
    print(f"Analysis result: {analysis}")
    
    # Example 3: Show how to use the tile detector directly
    print("\n3. Tile detection example:")
    detector = TileDetector()
    
    # Preprocess the image
    processed = detector.preprocess_image(test_image)
    
    # Find contours
    contours = detector.find_tile_contours(processed)
    print(f"Found {len(contours)} potential tile contours")
    
    # Draw contours on the image
    result_image = test_image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # Save the result
    cv2.imwrite('detected_tiles.jpg', result_image)
    print("Saved detected tiles image: detected_tiles.jpg")
    
    print("\n" + "=" * 30)
    print("Example completed!")
    print("\nNext steps:")
    print("1. Add tile template images for recognition")
    print("2. Implement actual tile recognition logic")
    print("3. Add more yaku patterns")
    print("4. Test with real mahjong tile images")


if __name__ == "__main__":
    main() 