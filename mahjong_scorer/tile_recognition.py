"""
Tile recognition module for mahjong tiles using template matching.
"""

import cv2
import numpy as np
import os
import glob
from typing import Dict, Optional, Tuple, List


class TileRecognizer:
    """Handles recognition of mahjong tiles using template matching."""
    
    def __init__(self, min_confidence: float = 0.7):
        """Initialize the tile recognizer."""
        self.tile_templates: Dict[str, np.ndarray] = {}
        self.min_confidence = min_confidence
        self.template_loaded = False
        
    def load_templates(self, template_dir: str = "templates") -> bool:
        """
        Load tile templates for recognition.
        
        Args:
            template_dir: Directory containing tile template images
            
        Returns:
            True if templates loaded successfully, False otherwise
        """
        if not os.path.exists(template_dir):
            print(f"Template directory not found: {template_dir}")
            return False
            
        self.tile_templates = {}
        
        # Load all PNG files in the template directory
        template_files = glob.glob(os.path.join(template_dir, "*.png"))
        
        if not template_files:
            print(f"No template files found in {template_dir}")
            return False
        
        for template_file in template_files:
            # Extract tile name from filename
            tile_name = os.path.splitext(os.path.basename(template_file))[0]
            
            # Load template image
            template_img = cv2.imread(template_file)
            if template_img is not None:
                # Convert to grayscale for template matching
                template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
                self.tile_templates[tile_name] = template_gray
                print(f"Loaded template: {tile_name}")
            else:
                print(f"Failed to load template: {template_file}")
        
        self.template_loaded = len(self.tile_templates) > 0
        print(f"Loaded {len(self.tile_templates)} tile templates")
        
        return self.template_loaded
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better template matching.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Normalize brightness and contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Normalize the image
        gray = cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
        
        return gray
    
    def recognize_tile(self, tile_image: np.ndarray) -> Optional[str]:
        """
        Recognize a single tile using template matching.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tile type (e.g., '1m', '2p', '3s', 'east', etc.) or None if unknown
        """
        if not self.template_loaded:
            print("Warning: No templates loaded. Call load_templates() first.")
            return None
        
        # Preprocess the tile image
        tile_processed = self.preprocess_image(tile_image)
        
        best_match = None
        best_confidence = 0.0
        
        # Try to match against all templates
        for tile_name, template in self.tile_templates.items():
            # Preprocess template
            template_processed = self.preprocess_image(template)
            
            # Resize template to match tile size if needed
            if template_processed.shape != tile_processed.shape:
                template_resized = cv2.resize(template_processed, 
                                            (tile_processed.shape[1], tile_processed.shape[0]))
            else:
                template_resized = template_processed
            
            # Perform template matching
            result = cv2.matchTemplate(tile_processed, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            confidence = max_val
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = tile_name
        
        # Return best match if confidence is above threshold
        if best_confidence >= self.min_confidence:
            return best_match
        else:
            return None
    
    def recognize_tile_with_confidence(self, tile_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize tile and return confidence score.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tuple of (tile_name, confidence) or (None, 0.0)
        """
        if not self.template_loaded:
            return None, 0.0
        
        tile_processed = self.preprocess_image(tile_image)
        
        best_match = None
        best_confidence = 0.0
        
        for tile_name, template in self.tile_templates.items():
            template_processed = self.preprocess_image(template)
            
            if template_processed.shape != tile_processed.shape:
                template_resized = cv2.resize(template_processed, 
                                            (tile_processed.shape[1], tile_processed.shape[0]))
            else:
                template_resized = template_processed
            
            result = cv2.matchTemplate(tile_processed, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            confidence = max_val
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = tile_name
        
        return best_match, best_confidence
    
    def recognize_multiple_tiles(self, tile_images: List[np.ndarray]) -> List[Tuple[Optional[str], float]]:
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
        return list(self.tile_templates.keys())
    
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
    
    def get_template_count(self) -> int:
        """
        Get the number of loaded templates.
        
        Returns:
            Number of loaded templates
        """
        return len(self.tile_templates) 