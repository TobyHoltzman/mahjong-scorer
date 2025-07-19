"""
Tile detection module for mahjong scorer using OpenCV.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from .tile_recognition import TileRecognizer


class TileDetector:
    """Detects mahjong tiles using computer vision."""
    
    def __init__(self):
        """Initialize the tile detector."""
        self.recognizer = TileRecognizer()
        self.detection_params = {
            'min_contour_area': 1000,
            'max_contour_area': 50000,
            'min_aspect_ratio': 0.5,
            'max_aspect_ratio': 2.0,
        }
        
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
            if (self.detection_params['min_contour_area'] < area < 
                self.detection_params['max_contour_area']):
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if (self.detection_params['min_aspect_ratio'] < aspect_ratio < 
                    self.detection_params['max_aspect_ratio']):
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
        Recognize the type of mahjong tile using the recognizer.
        
        Args:
            tile_image: Image of a single tile
            
        Returns:
            Tile type (e.g., '1m', '2p', '3s', 'east', etc.) or None if unknown
        """
        return self.recognizer.recognize_tile(tile_image)
    
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
    
    def load_tile_templates(self, template_dir: str = "templates") -> bool:
        """
        Load tile templates for recognition.
        
        Args:
            template_dir: Directory containing tile template images
            
        Returns:
            True if templates loaded successfully
        """
        return self.recognizer.load_templates(template_dir)
    
    def calibrate(self, calibration_image: np.ndarray):
        """
        Calibrate the detector using a known calibration image.
        
        Args:
            calibration_image: Image with known tile layout
        """
        # TODO: Implement calibration
        # This would help adjust detection parameters
        pass
    
    def set_detection_parameters(self, **kwargs):
        """
        Set detection parameters.
        
        Args:
            **kwargs: Parameters to update (min_contour_area, max_contour_area, etc.)
        """
        for key, value in kwargs.items():
            if key in self.detection_params:
                self.detection_params[key] = value
    
    def get_detection_parameters(self) -> dict:
        """
        Get current detection parameters.
        
        Returns:
            Dictionary of current detection parameters
        """
        return self.detection_params.copy()
    
    def set_confidence_threshold(self, threshold: float):
        """
        Set the confidence threshold for recognition.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.recognizer.set_confidence_threshold(threshold)
    
    def get_available_tiles(self) -> List[str]:
        """
        Get list of available tile types.
        
        Returns:
            List of tile names that can be recognized
        """
        return self.recognizer.get_available_tiles() 