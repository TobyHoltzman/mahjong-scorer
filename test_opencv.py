#!/usr/bin/env python3
"""
Test script to verify OpenCV installation and basic functionality.
"""

import cv2
import numpy as np
import sys

def test_opencv_installation():
    """Test if OpenCV is properly installed and working."""
    print("Testing OpenCV installation...")
    
    # Check OpenCV version
    print(f"OpenCV version: {cv2.getVersionMajor()}")
    
    # Test basic functionality
    try:
        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White rectangle
        
        # Test basic operations
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print("✓ Basic OpenCV operations working")
        print("✓ Image processing functions available")
        
        # Test if we can read/write images (if we have a camera, we could test video too)
        print("✓ OpenCV installation successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing OpenCV: {e}")
        return False

def test_mahjong_related_functions():
    """Test functions that might be useful for mahjong tile recognition."""
    print("\nTesting mahjong-related OpenCV functions...")
    
    try:
        # Test contour detection (useful for tile detection)
        test_image = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
        contours, _ = cv2.findContours(test_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"✓ Contour detection working (found {len(contours)} contours)")
        
        # Test template matching (useful for tile recognition)
        template = np.zeros((50, 50), dtype=np.uint8)
        cv2.rectangle(template, (10, 10), (40, 40), 255, -1)
        result = cv2.matchTemplate(test_image, template, cv2.TM_CCOEFF_NORMED)
        print("✓ Template matching working")
        
        # Test color detection (useful for tile color analysis)
        color_image = np.zeros((100, 100, 3), dtype=np.uint8)
        color_image[:, :] = [0, 255, 0]  # Green
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        print("✓ Color space conversion working")
        
        print("✓ All mahjong-related functions available!")
        return True
        
    except Exception as e:
        print(f"✗ Error testing mahjong functions: {e}")
        return False

if __name__ == "__main__":
    print("Mahjong Scorer - OpenCV Test")
    print("=" * 40)
    
    success = True
    success &= test_opencv_installation()
    success &= test_mahjong_related_functions()
    
    if success:
        print("\n🎉 All tests passed! OpenCV is ready for mahjong tile recognition.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run this test again: python test_opencv.py")
        print("3. Start building your mahjong tile recognition system!")
    else:
        print("\n❌ Some tests failed. Please check your OpenCV installation.")
        sys.exit(1) 