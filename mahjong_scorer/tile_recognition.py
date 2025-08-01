"""
Tile recognition module for mahjong tiles using CNN with PyTorch.
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, Optional, Tuple, List

# Import PyTorch components
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class MahjongTileCNN(nn.Module):
    """CNN architecture for mahjong tile recognition."""
    
    def __init__(self, num_classes: int = 34):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of tile classes (34 for standard mahjong)
        """
        super(MahjongTileCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Assuming 64x64 input -> 4x4 after pooling
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class TileRecognizer:
    """Handles recognition of mahjong tiles using CNN with PyTorch."""
    
    def __init__(self, min_confidence: float = 0.7, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the tile recognizer.
        
        Args:
            min_confidence: Minimum confidence threshold
            model_path: Path to pre-trained CNN model
            device: Device to use for PyTorch ('cpu', 'cuda', or 'auto')
        """
        self.min_confidence = min_confidence
        self.device = self._get_device(device)
        self.model = None
        self.class_names = []
        self.transform = self._get_transforms()
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def _get_device(self, device: str):
        """Get the appropriate device for PyTorch."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _get_transforms(self):
        """Get image transformations for the model."""
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for the CNN model.
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transformations
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a pre-trained CNN model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Load model architecture and weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            num_classes = checkpoint.get('num_classes', 34)
            self.model = MahjongTileCNN(num_classes=num_classes)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Load class names
            self.class_names = checkpoint.get('class_names', [])
            
            print(f"CNN model loaded successfully from {model_path}")
            print(f"Device: {self.device}")
            print(f"Number of classes: {num_classes}")
            
            return True
            
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def save_model(self, model_path: str, class_names: List[str]) -> bool:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
            class_names: List of class names
            
        Returns:
            True if model saved successfully
        """
        try:
            if self.model is None:
                print("No model to save")
                return False
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'num_classes': len(class_names),
                'class_names': class_names,
                'architecture': 'MahjongTileCNN'
            }
            
            torch.save(checkpoint, model_path)
            print(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def recognize_tile(self, tile_image: np.ndarray) -> Optional[str]:
        """
        Recognize a single tile using the CNN model.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tile type (e.g., '1m', '2p', '3s', 'east', etc.) or None if unknown
        """
        if self.model is None:
            print("Warning: No model loaded. Call load_model() first.")
            return None
        
        # Get prediction with confidence
        tile_name, confidence = self.recognize_tile_with_confidence(tile_image)
        
        # Return result if confidence is above threshold
        if confidence >= self.min_confidence:
            return tile_name
        else:
            return None
    
    def recognize_tile_with_confidence(self, tile_image: np.ndarray) -> Tuple[str, float]:
        """
        Recognize tile and return confidence score.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tuple of (tile_name, confidence) or (None, 0.0)
        """
        if self.model is None:
            return '', 0.0
        
        try:
            # Preprocess image
            tensor = self._preprocess_image(tile_image)
            tensor = tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
            
            # Get class name
            if 0 <= predicted_idx < len(self.class_names):
                tile_name = self.class_names[predicted_idx]
            else:
                tile_name = ''
                confidence = 0.0
            
            return tile_name, confidence
            
        except Exception as e:
            print(f"Error during recognition: {e}")
            return '', 0.0
    
    def recognize_multiple_tiles(self, tile_images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Recognize multiple tiles at once.
        
        Args:
            tile_images: List of tile images
            
        Returns:
            List of (tile_name, confidence) tuples
        """
        results = []
        for tile_image in tile_images:
            tile_name, confidence = self.recognize_tile_with_confidence(tile_image)
            results.append((tile_name, confidence))
        return results
    
    def get_available_tiles(self) -> List[str]:
        """
        Get list of available tile types.
        
        Returns:
            List of tile names that can be recognized
        """
        return self.class_names.copy()
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for recognition.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.min_confidence = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "CNN model loaded",
            "device": str(self.device),
            "architecture": "MahjongTileCNN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_threshold": self.min_confidence
        }
    
    def visualize_prediction(self, tile_image: np.ndarray, save_path: Optional[str] = None):
        """
        Visualize the model's prediction on a tile image.
        
        Args:
            tile_image: Image of a single tile
            save_path: Optional path to save the visualization
        """
        if self.model is None:
            print("No model loaded for visualization")
            return
        
        # Get prediction
        tile_name, confidence = self.recognize_tile_with_confidence(tile_image)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        rgb_image = cv2.cvtColor(tile_image, cv2.COLOR_BGR2RGB)
        ax1.imshow(rgb_image)
        ax1.set_title(f"Input Tile Image")
        ax1.axis('off')
        
        # Prediction results
        if tile_name:
            ax2.text(0.1, 0.6, f"Predicted: {tile_name}", fontsize=14, fontweight='bold')
            ax2.text(0.1, 0.4, f"Confidence: {confidence:.3f}", fontsize=12)
            ax2.text(0.1, 0.2, f"Threshold: {self.min_confidence:.3f}", fontsize=12)
        else:
            ax2.text(0.1, 0.5, "No confident prediction", fontsize=14, color='red')
            ax2.text(0.1, 0.3, f"Best confidence: {confidence:.3f}", fontsize=12)
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show() 