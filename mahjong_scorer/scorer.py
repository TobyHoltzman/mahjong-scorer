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
        Check if a hand forms a valid winning pattern (14 tiles).
        Must be composed of 4 groups (kan/triplets/sequences) + 1 pair,
        or seven pairs, or thirteen orphans.
        
        Args:
            tiles: List of tile strings
            
        Returns:
            True if valid winning hand, False otherwise
        """
        # Check for exactly 14 tiles
        if len(tiles) < 14:
            return False
            
        # Check tile counts (no more than 4 of each)
        tile_counts = self.parse_tiles(tiles)
        if any(count > 4 for count in tile_counts.values()):
            return False
            
        # Check for thirteen orphans (kokushi musou)
        terminals = ['1m', '9m', '1p', '9p', '1s', '9s'] + self.honors
        if self._is_thirteen_orphans(tiles, terminals):
            return True
            
        # Check for seven pairs (chiitoitsu)
        if self._is_seven_pairs(tiles):
            return True
            
        # Check for standard hand (4 groups + pair)
        return self._is_standard_win(tiles)
        
    def _is_thirteen_orphans(self, tiles: List[str], terminals: List[str]) -> bool:
        """Check if hand is thirteen orphans (kokushi musou)."""
        # Must contain all terminals and honors
        unique_tiles = set(tiles)
        if not all(t in unique_tiles for t in terminals):
            return False
            
        # Must have exactly one duplicate from terminals/honors
        tile_counts = self.parse_tiles(tiles)
        doubles = sum(1 for t in terminals if tile_counts.get(t, 0) == 2)
        singles = sum(1 for t in terminals if tile_counts.get(t, 0) == 1)
        
        return doubles == 1 and singles == 12
        
    def _is_seven_pairs(self, tiles: List[str]) -> bool:
        """Check if hand is seven pairs (chiitoitsu)."""
        tile_counts = self.parse_tiles(tiles)
        pairs = sum(1 for count in tile_counts.values() if count == 2)
        return pairs == 7 and len(tile_counts) == 7
        
    def _is_standard_win(self, tiles: List[str]) -> bool:
        """Check if tiles form a standard winning hand (4 groups + pair)."""
        tile_counts = self.parse_tiles(tiles)
        
        # Try each tile type as the pair
        for tile, count in tile_counts.items():
            if count >= 2:
                # Remove pair and check if remaining tiles form valid groups
                remaining = tiles.copy()
                remaining.remove(tile)
                remaining.remove(tile)
                groups = self._can_form_groups(remaining)
                if groups == 4:
                    return True
        
        return False
        
    def _can_form_groups(self, tiles: List[str]) -> int:
        """
        Check if tiles can be arranged into 4 groups (kan/triplets/sequences).
        A group can be:
        - Kan (four of a kind)
        - Triplet (three of a kind)
        - Sequence (three consecutive numbers in same suit)
        """
        if not tiles:
            return 0
            
        tile_counts = self.parse_tiles(tiles)
        first_tile = tiles[0]
        
        # Try forming a kan (four of a kind)
        if tile_counts[first_tile] >= 4:
            remaining = tiles.copy()
            for _ in range(4):
                remaining.remove(first_tile)
            groups = self._can_form_groups(remaining)
            if groups >= 0:
                return 1 + groups
                
        # Try forming a triplet
        if tile_counts[first_tile] >= 3:
            remaining = tiles.copy()
            for _ in range(3):
                remaining.remove(first_tile)
            groups = self._can_form_groups(remaining)
            if groups >= 0:
                return 1 + groups
                
        # Try forming a sequence if it's a numbered tile
        if first_tile[-1] in self.suits:  # is a suited tile
            number = int(first_tile[0])
            suit = first_tile[-1]
            next_tile = f"{number+1}{suit}"
            next_next_tile = f"{number+2}{suit}"
            
            # Check if we can form a sequence
            if (number <= 7 and  # sequence won't exceed 9
                next_tile in tile_counts and
                next_next_tile in tile_counts):
                # Remove sequence and recurse
                remaining = tiles.copy()
                remaining.remove(first_tile)
                remaining.remove(next_tile)
                remaining.remove(next_next_tile)
                groups = self._can_form_groups(remaining)
                if groups >= 0:
                    return 1 + groups
                    
        return -1
    
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