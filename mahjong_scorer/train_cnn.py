"""
Training script for the CNN-based mahjong tile recognizer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Dict
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from tile_recognition import MahjongTileCNN


class MahjongTileDataset(Dataset):
    """Dataset for mahjong tile images."""
    
    def __init__(self, data_dir: str, transform=None, generate_synthetic: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing tile images organized by class
            transform: Image transformations
            generate_synthetic: Whether to generate synthetic data if real data is insufficient
        """
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = []
        self.image_paths = []
        self.class_to_idx = {}
        
        # Load real data if available
        if os.path.exists(data_dir):
            self._load_real_data()
        
        # Generate synthetic data if needed
        if generate_synthetic and len(self.image_paths) < 1000:  # Arbitrary threshold
            print("Generating synthetic training data...")
            self._generate_synthetic_data()
        
        print(f"Dataset loaded: {len(self.image_paths)} images, {len(self.class_names)} classes")
    
    def _load_real_data(self):
        """Load real tile images from the data directory."""
        for class_name in sorted(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = len(self.class_names)
                self.class_names.append(class_name)
                
                # Load images from this class
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append((img_path, self.class_to_idx[class_name]))
    
    def _generate_synthetic_data(self):
        """Generate synthetic tile images for training."""
        # Define mahjong tile classes
        tile_classes = [
            # Man tiles (1-9)
            '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
            # Pin tiles (1-9)
            '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
            # Sou tiles (1-9)
            '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
            # Honor tiles
            'east', 'south', 'west', 'north',
            'red', 'green', 'white'
        ]
        
        # Generate synthetic images for each class
        for class_name in tile_classes:
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_names)
                self.class_names.append(class_name)
            
            class_idx = self.class_to_idx[class_name]
            
            # Generate multiple variations of each tile
            for i in range(50):  # 50 synthetic images per class
                synthetic_img = self._create_synthetic_tile(class_name)
                self.image_paths.append((synthetic_img, class_idx))
    
    def _create_synthetic_tile(self, tile_type: str) -> np.ndarray:
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
            if tile_type in ['east', 'south', 'west', 'north']:
                number = tile_type[0].upper()
            elif tile_type in ['red', 'green', 'white']:
                number = tile_type[0].upper()
            else:
                number = tile_type[0].upper() if len(tile_type) > 0 else 'X'
        
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
        
        # Add some noise and variations
        noise = np.random.normal(0, 10, tile.shape).astype(np.uint8)
        tile = np.clip(tile + noise, 0, 255)
        
        # Add slight rotation
        angle = random.uniform(-5, 5)
        center = (32, 32)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        tile = cv2.warpAffine(tile, rotation_matrix, (64, 64))
        
        return tile
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.image_paths[idx]
        
        if isinstance(img_path, str) and os.path.exists(img_path):
            # Load real image
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to synthetic image if loading fails
                image = self._create_synthetic_tile("1m")  # Default fallback
        else:
            # Use synthetic image (img_path is actually the image array)
            image = img_path
        
        # Convert to PIL Image (ensure image is numpy array)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        else:
            # Fallback for any other type
            image = Image.fromarray(np.array(image))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx


def train_cnn_model(data_dir: str = "training_data", 
                   model_save_path: str = "models/mahjong_cnn.pth",
                   num_epochs: int = 50,
                   batch_size: int = 32,
                   learning_rate: float = 0.001,
                   device: str = "auto"):
    """
    Train the CNN model for mahjong tile recognition.
    
    Args:
        data_dir: Directory containing training data
        model_save_path: Path to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        device: Device to use for training
    """
    # Set up device
    if device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    
    print(f"Training on device: {device_obj}")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = MahjongTileDataset(data_dir, transform=transform, generate_synthetic=False)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    num_classes = len(dataset.class_names)
    model = MahjongTileCNN(num_classes=num_classes)
    model.to(device_obj)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)  # Slower decay
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device_obj), labels.to(device_obj)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_pbar:
                images, labels = images.to(device_obj), labels.to(device_obj)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print()
    
    # Save the trained model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'class_names': dataset.class_names,
        'architecture': 'MahjongTileCNN',
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    }
    
    torch.save(checkpoint, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model, dataset.class_names


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Train the model
    model, class_names = train_cnn_model(
        data_dir="training_data",
        model_save_path="models/mahjong_cnn.pth",
        num_epochs=60,
        batch_size=16,  # Smaller batch size for better generalization with limited data
        learning_rate=0.001
    )
    
    print("Training completed!")
    print(f"Model can recognize {len(class_names)} tile types:")
    for i, class_name in enumerate(class_names):
        print(f"  {i+1:2d}. {class_name}")