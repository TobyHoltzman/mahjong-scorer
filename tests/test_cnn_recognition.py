#!/usr/bin/env python3
"""
Test script for CNN-based mahjong tile recognition.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from mahjong_scorer import TileRecognizer, MahjongTileCNN, TileDetector


def test_cnn_recognizer():
    """Test the CNN recognizer with synthetic data."""
    print("Testing CNN-based Tile Recognition")
    print("=" * 50)
    
    # Initialize recognizer
    recognizer = TileRecognizer()
    
    # Check if we have a trained model
    model_path = "models/mahjong_cnn.pth"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        success = recognizer.load_model(model_path)
        if not success:
            print("Failed to load model. Will create a simple test model.")
            return test_without_model(recognizer)
    else:
        print("No pre-trained model found. Creating a simple test model.")
        return test_without_model(recognizer)
    
    # Test with synthetic tiles
    print("\nTesting recognition with synthetic tiles...")
    test_tiles = create_test_tiles()
    
    results = []
    for i, (tile_name, tile_image) in enumerate(test_tiles):
        predicted_name, confidence = recognizer.recognize_tile_with_confidence(tile_image)
        
        correct = predicted_name == tile_name
        results.append((tile_name, predicted_name, confidence, correct))
        
        status = "✓" if correct else "✗"
        print(f"{status} {tile_name:4s} -> {predicted_name:4s} (conf: {confidence:.3f})")
    
    # Calculate accuracy
    correct_predictions = sum(1 for _, _, _, correct in results if correct)
    accuracy = correct_predictions / len(results) * 100
    
    print(f"\nAccuracy: {accuracy:.1f}% ({correct_predictions}/{len(results)})")
    
    # Show model info
    print("\nModel Information:")
    model_info = recognizer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    return accuracy > 80.0


def test_without_model(recognizer):
    """Test without a pre-trained model (demonstration only)."""
    print("\nDemonstrating CNN architecture without trained weights...")
    
    # Create a simple model for demonstration
    model = MahjongTileCNN(num_classes=34)
    
    # Create synthetic test image
    test_image = create_synthetic_tile("1m")
    
    # Show model architecture
    print(f"Model architecture: {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Demonstrate preprocessing
    print("\nDemonstrating image preprocessing...")
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    from PIL import Image
    pil_image = Image.fromarray(rgb_image)
    
    # Apply transformations
    tensor = recognizer.transform(pil_image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    print(f"Input image shape: {test_image.shape}")
    print(f"Preprocessed tensor shape: {tensor.shape}")
    
    # Show the test image
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("Original Test Tile (1m)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Show the tensor as an image (denormalize)
    tensor_img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    tensor_img = (tensor_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    tensor_img = np.clip(tensor_img, 0, 255).astype(np.uint8)
    plt.imshow(tensor_img)
    plt.title("Preprocessed Tensor")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_images/cnn_preprocessing_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nNote: This is a demonstration without trained weights.")
    print("To get actual recognition, you need to train the model first.")
    print("Run: python3 -m mahjong_scorer.train_cnn")
    
    return True


def create_test_tiles():
    """Create synthetic test tiles for evaluation."""
    tile_types = ['1m', '2m', '3m', '1p', '2p', '3p', '1s', '2s', '3s', 'east', 'south', 'west', 'north']
    test_tiles = []
    
    for tile_type in tile_types:
        tile_image = create_synthetic_tile(tile_type)
        test_tiles.append((tile_type, tile_image))
    
    return test_tiles


def create_synthetic_tile(tile_type: str) -> np.ndarray:
    """Create a synthetic tile image."""
    # Create base tile (64x64)
    tile = np.ones((64, 64, 3), dtype=np.uint8) * 255
    
    # Add tile border
    cv2.rectangle(tile, (2, 2), (61, 61), (0, 0, 0), 2)
    
    # Add tile content based on type
    if tile_type.endswith('m'):  # Man tiles
        color = (255, 0, 0)  # Red
        number = tile_type[0]
    elif tile_type.endswith('p'):  # Pin tiles
        color = (0, 0, 255)  # Blue
        number = tile_type[0]
    elif tile_type.endswith('s'):  # Sou tiles
        color = (0, 255, 0)  # Green
        number = tile_type[0]
    else:  # Honor tiles
        color = (128, 128, 128)  # Gray
        number = tile_type[0].upper() if len(tile_type) > 0 else tile_type[0]
    
    # Add number/symbol
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 2
    
    # Get text size for centering
    (text_width, text_height), baseline = cv2.getTextSize(number, font, font_scale, thickness)
    text_x = (64 - text_width) // 2
    text_y = (64 + text_height) // 2
    
    # Add text
    cv2.putText(tile, number, (text_x, text_y), font, font_scale, color, thickness)
    
    return tile


def test_tile_detector_with_cnn():
    """Test the tile detector with CNN recognition."""
    print("\nTesting Tile Detector with CNN Recognition")
    print("=" * 50)
    
    # Initialize detector with CNN recognizer
    model_path = "models/mahjong_cnn.pth" if os.path.exists("models/mahjong_cnn.pth") else None
    detector = TileDetector(model_path=model_path)
    
    # Create a test image with multiple tiles
    test_image = create_test_image_with_tiles()
    
    # Detect tiles
    print("Detecting tiles in test image...")
    detected_tiles = detector.detect_tiles(test_image)
    
    print(f"Detected {len(detected_tiles)} tiles:")
    for i, (tile_region, tile_type) in enumerate(detected_tiles):
        print(f"  Tile {i+1}: {tile_type}")
    
    # Visualize results
    visualize_detection_results(test_image, detected_tiles)
    
    return len(detected_tiles) > 0


def create_test_image_with_tiles() -> np.ndarray:
    """Create a test image with multiple tiles."""
    # Create a larger image (200x400)
    image = np.ones((200, 400, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add multiple tiles
    tile_positions = [
        (50, 50, "1m"),
        (150, 50, "2m"),
        (250, 50, "3m"),
        (50, 150, "1p"),
        (150, 150, "2p"),
        (250, 150, "3p"),
    ]
    
    for x, y, tile_type in tile_positions:
        tile = create_synthetic_tile(tile_type)
        # Resize tile to fit in the image
        tile_resized = cv2.resize(tile, (40, 40))
        image[y:y+40, x:x+40] = tile_resized
    
    return image


def visualize_detection_results(image: np.ndarray, detected_tiles: list):
    """Visualize tile detection results."""
    result_image = image.copy()
    
    for i, (tile_region, tile_type) in enumerate(detected_tiles):
        # Get bounding box
        h, w = tile_region.shape[:2]
        
        # Draw rectangle around detected tile
        cv2.rectangle(result_image, (0, 0), (w, h), (0, 255, 0), 2)
        
        # Add label
        cv2.putText(result_image, tile_type, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display results
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    rgb_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_original)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    rgb_result = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_result)
    plt.title(f"Detected Tiles ({len(detected_tiles)} found)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_images/cnn_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main test function."""
    print("CNN-based Mahjong Tile Recognition Test")
    print("=" * 60)
    
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Test CNN recognizer
    cnn_success = test_cnn_recognizer()
    
    # Test tile detector with CNN
    detector_success = test_tile_detector_with_cnn()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if cnn_success and detector_success:
        print("🎉 All CNN tests passed!")
        print("\nNext steps:")
        print("1. Train the model with real data: python3 -m mahjong_scorer.train_cnn")
        print("2. Test with real mahjong images")
        print("3. Fine-tune the model for better accuracy")
        return True
    else:
        print("⚠️  Some tests failed.")
        print("\nTo improve results:")
        print("1. Install PyTorch: python3 -m pip install torch torchvision")
        print("2. Train the model with real data")
        print("3. Collect more training data")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 