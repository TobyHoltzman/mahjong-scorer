#!/usr/bin/env python3
"""
Test script for the CNN training functionality.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from mahjong_scorer import MahjongTileCNN
from mahjong_scorer.train_cnn import MahjongTileDataset, train_cnn_model


def test_cnn_architecture():
    """Test the CNN architecture."""
    print("Testing CNN Architecture")
    print("=" * 40)
    
    # Test model creation
    model = MahjongTileCNN(num_classes=34)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 64, 64)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, 34)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return output.shape == (batch_size, 34)


def test_dataset():
    """Test the dataset creation."""
    print("\nTesting Dataset Creation")
    print("=" * 40)
    
    # Create dataset
    dataset = MahjongTileDataset("test_data", generate_synthetic=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.class_names)}")
    print(f"Classes: {dataset.class_names[:10]}...")  # Show first 10
    
    # Test getting an item
    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"Image type: {type(image)}")
        print(f"Label: {label}")
        print(f"Label class: {dataset.class_names[label]}")
    
    return len(dataset) > 0


def test_synthetic_tile_creation():
    """Test synthetic tile creation."""
    print("\nTesting Synthetic Tile Creation")
    print("=" * 40)
    
    dataset = MahjongTileDataset("test_data", generate_synthetic=False)
    
    # Test different tile types
    tile_types = ['1m', '5p', '9s', 'east', 'red']
    
    for tile_type in tile_types:
        tile = dataset._create_synthetic_tile(tile_type)
        print(f"{tile_type}: shape {tile.shape}, dtype {tile.dtype}")
        
        # Check if tile is valid
        if tile.shape == (64, 64, 3) and tile.dtype == np.uint8:
            print(f"  ✓ {tile_type} tile created successfully")
        else:
            print(f"  ✗ {tile_type} tile creation failed")
            return False
    
    return True


def test_training_function():
    """Test the training function (short version)."""
    print("\nTesting Training Function")
    print("=" * 40)
    
    try:
        # Test with minimal parameters
        model, class_names = train_cnn_model(
            data_dir="test_training_data",
            model_save_path="test_models/test_model.pth",
            num_epochs=2,  # Very short training
            batch_size=8,   # Small batch size
            learning_rate=0.001
        )
        
        print(f"Training completed successfully!")
        print(f"Model can recognize {len(class_names)} tile types")
        
        # Check if model file was created
        if os.path.exists("test_models/test_model.pth"):
            print("✓ Model file saved successfully")
            return True
        else:
            print("✗ Model file not found")
            return False
            
    except Exception as e:
        print(f"Training failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    import shutil
    
    # Remove test directories
    test_dirs = ["test_data", "test_training_data", "test_models"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned up: {test_dir}")


def main():
    """Main test function."""
    print("CNN Training Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("CNN Architecture", test_cnn_architecture),
        ("Dataset Creation", test_dataset),
        ("Synthetic Tile Creation", test_synthetic_tile_creation),
        ("Training Function", test_training_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Training functionality is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 