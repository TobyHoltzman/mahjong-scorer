"""
Tile detection module for mahjong scorer using OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class TileDetector:
    """Detects and recognizes mahjong tiles using computer vision."""
    
    def __init__(self):
        """Initialize the tile detector."""
        self.tile_templates = {}  # Will store tile templates for matching
        self.min_confidence = 0.7
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better tile detection.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def find_tile_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Find contours that might represent mahjong tiles.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of contours that could be tiles
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        tile_contours = []
        for contour in contours:
            # Filter contours by area and aspect ratio
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.5 < aspect_ratio < 2.0:  # Reasonable tile aspect ratio
                    tile_contours.append(contour)
        
        return tile_contours
    
    def extract_tile_region(self, image: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """
        Extract the region of a detected tile.
        
        Args:
            image: Original image
            contour: Tile contour
            
        Returns:
            Extracted tile region
        """
        x, y, w, h = cv2.boundingRect(contour)
        return image[y:y+h, x:x+w]
    
    def recognize_tile(self, tile_image: np.ndarray) -> Optional[str]:
        """
        Recognize the type of mahjong tile.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tile type (e.g., '1m', '2p', '3s', 'east', etc.) or None if unknown
        """
        # This is a placeholder - you'll need to implement actual tile recognition
        # using template matching, machine learning, or other CV techniques
        
        # For now, return None to indicate unknown tile
        return None
    
    def detect_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, str]]:
        """
        Detect and recognize all tiles in an image.
        
        Args:
            image: Input image containing mahjong tiles
            
        Returns:
            List of (tile_region, tile_type) tuples
        """
        # Preprocess the image
        processed = self.preprocess_image(image)
        
        # Find tile contours
        contours = self.find_tile_contours(processed)
        
        # Extract and recognize tiles
        detected_tiles = []
        for contour in contours:
            tile_region = self.extract_tile_region(image, contour)
            tile_type = self.recognize_tile(tile_region)
            
            if tile_type:
                detected_tiles.append((tile_region, tile_type))
        
        return detected_tiles
    
    def load_tile_templates(self, template_dir: str):
        """
        Load tile templates for recognition.
        
        Args:
            template_dir: Directory containing tile template images
        """
        # TODO: Implement template loading
        # This would load reference images for each tile type
        pass
    
    def calibrate(self, calibration_image: np.ndarray):
        """
        Calibrate the detector using a known calibration image.
        
        Args:
            calibration_image: Image with known tile layout
        """
        # TODO: Implement calibration
        # This would help adjust detection parameters
        pass 