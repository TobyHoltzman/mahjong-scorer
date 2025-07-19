"""
Mahjong scoring module for Riichi mahjong.
"""

from typing import List, Dict, Tuple, Any
from .tile_detector import TileDetector


class MahjongScorer:
    """Calculates scores for Riichi mahjong hands."""
    
    def __init__(self):
        """Initialize the mahjong scorer."""
        self.tile_detector = TileDetector()
        
        # Define tile types
        self.suits = ['m', 'p', 's']  # man, pin, sou
        self.honors = ['east', 'south', 'west', 'north', 'red', 'green', 'white']
        self.numbers = list(range(1, 10))  # 1-9
        
    def parse_tiles(self, tile_list: List[str]) -> Dict[str, int]:
        """
        Parse a list of tile strings into a tile count dictionary.
        
        Args:
            tile_list: List of tile strings (e.g., ['1m', '2m', '3m'])
            
        Returns:
            Dictionary with tile counts
        """
        tile_counts = {}
        
        for tile in tile_list:
            if tile in tile_counts:
                tile_counts[tile] += 1
            else:
                tile_counts[tile] = 1
                
        return tile_counts
    
    def is_valid_hand(self, tiles: List[str]) -> bool:
        """
        Check if a hand is valid (13 tiles).
        
        Args:
            tiles: List of tile strings
            
        Returns:
            True if valid, False otherwise
        """
        return len(tiles) == 13
    
    def count_yaku(self, tiles: List[str]) -> List[Tuple[str, int]]:
        """
        Count yaku (scoring patterns) in a hand.
        
        Args:
            tiles: List of tile strings
            
        Returns:
            List of (yaku_name, han_value) tuples
        """
        yaku_list = []
        tile_counts = self.parse_tiles(tiles)
        
        # Check for basic yaku (this is a simplified version)
        
        # Tanyao (all simples)
        if self._is_tanyao(tiles):
            yaku_list.append(("Tanyao", 1))
        
        # Yakuhai (dragon/honor tiles)
        yakuhai_count = self._count_yakuhai(tiles)
        if yakuhai_count > 0:
            yaku_list.append(("Yakuhai", yakuhai_count))
        
        # TODO: Add more yaku patterns
        
        return yaku_list
    
    def _is_tanyao(self, tiles: List[str]) -> bool:
        """Check if hand contains only simple tiles (2-8)."""
        for tile in tiles:
            if tile in self.honors:  # Honor tiles
                return False
            if tile.endswith(('1', '9')):  # Terminal tiles
                return False
        return True
    
    def _count_yakuhai(self, tiles: List[str]) -> int:
        """Count yakuhai (dragon/honor tiles)."""
        count = 0
        tile_counts = self.parse_tiles(tiles)
        
        for honor in self.honors:
            if honor in tile_counts and tile_counts[honor] >= 3:
                count += 1
                
        return count
    
    def calculate_score(self, tiles: List[str], han: int, fu: int = 30) -> Dict[str, Any]:
        """
        Calculate the score for a hand.
        
        Args:
            tiles: List of tile strings
            han: Number of han
            fu: Number of fu (default 30)
            
        Returns:
            Dictionary with score information
        """
        if not self.is_valid_hand(tiles):
            return {"error": "Invalid hand size"}
        
        # Calculate base points
        if han >= 13:
            base_points = 8000  # Kazoe yakuman
        elif han >= 11:
            base_points = 6000  # Sanbaiman
        elif han >= 8:
            base_points = 4000  # Baiman
        elif han >= 6:
            base_points = 3000  # Haneman
        elif han >= 5:
            base_points = 2000  # Mangan
        else:
            # Calculate based on han and fu
            base_points = min(fu * (2 ** (han + 2)), 2000)
        
        # Calculate payments (simplified)
        payments = {
            "ron": base_points,
            "tsumo": {
                "dealer": base_points * 2,
                "non_dealer": base_points
            }
        }
        
        return {
            "base_points": base_points,
            "han": han,
            "fu": fu,
            "payments": payments
        }
    
    def analyze_hand_from_image(self, image) -> Dict:
        """
        Analyze a mahjong hand from an image.
        
        Args:
            image: OpenCV image array
            
        Returns:
            Dictionary with analysis results
        """
        # Detect tiles in the image
        detected_tiles = self.tile_detector.detect_tiles(image)
        
        # Extract tile types
        tile_types = [tile_type for _, tile_type in detected_tiles]
        
        # Count yaku
        yaku_list = self.count_yaku(tile_types)
        
        # Calculate total han
        total_han = sum(han for _, han in yaku_list)
        
        # Calculate score
        score_info = self.calculate_score(tile_types, total_han)
        
        return {
            "detected_tiles": tile_types,
            "yaku": yaku_list,
            "total_han": total_han,
            "score": score_info
        } 