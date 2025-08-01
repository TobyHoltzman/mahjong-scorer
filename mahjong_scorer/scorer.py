"""
Mahjong scoring module for Riichi mahjong.
"""

from typing import List, Dict, Tuple, Any, Optional
from .tile_detector import TileDetector
from .tile_recognition import TileRecognizer


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
        if not self.is_valid_hand(tiles):
            return []
            
        yaku_list = []
        tile_counts = self.parse_tiles(tiles)
        
        # Check for special patterns first (these are usually mutually exclusive)
        
        # Kokushi Musou (Thirteen Orphans)
        terminals = ['1m', '9m', '1p', '9p', '1s', '9s'] + self.honors
        if self._is_thirteen_orphans(tiles, terminals):
            return [("Kokushi Musou", 13)]
            
        # Chiitoitsu (Seven Pairs)
        if self._is_seven_pairs(tiles):
            yaku_list.append(("Chiitoitsu", 2))

        # Tanyao (all simples)
        if self._check_tanyao(tiles):
            yaku_list.append(("Tanyao", 1))
            
        # Yakuhai (valuable tiles)
        yakuhai_count = self._count_yakuhai(tiles)
        if yakuhai_count > 0:
            yaku_list.append(("Yakuhai", yakuhai_count))
            
        # Pinfu (no points hand)
        if self._check_pinfu(tiles):
            yaku_list.append(("Pinfu", 1))
            
        # Iipeikou (pure double sequence)
        if self._check_iipeikou(tiles):
            yaku_list.append(("Iipeikou", 1))
            
        # Sanshoku (three color straight)
        if self._check_sanshoku(tiles):
            yaku_list.append(("Sanshoku Doujun", 2))
            
        # Ittsuu (pure straight)
        if self._check_ittsuu(tiles):
            yaku_list.append(("Ittsuu", 2))
            
        # Honitsu (one suit plus honors)
        suit_type = self._check_honitsu(tiles)
        if suit_type:
            yaku_list.append(("Honitsu", 3))
            
        # Chinitsu (pure one suit, no honors)
        if self._check_chinitsu(tiles):
            yaku_list.append(("Chinitsu", 6))
            
        # Check for multiple kans
        kan_count = sum(1 for t, c in tile_counts.items() if c == 4)
        if kan_count >= 3:
            yaku_list.append(("Toitoi", 2))  # All triplets/kans
            yaku_list.append(("San Kantsu", 2))  # Three kans
        elif kan_count == 2:
            yaku_list.append(("Ryan Kantsu", 2))  # Two kans
            
        return yaku_list
        
    def _check_pinfu(self, tiles: List[str]) -> bool:
        """
        Check for pinfu (no points hand).
        Requirements:
        - Four sequences
        - Pair is not dragons or seat/prevalent wind
        - No triplets/kans
        - Must have a two-sided wait
        """
        # Get tile counts
        tile_counts = self.parse_tiles(tiles)
        
        # No triplets/kans allowed
        if any(count >= 3 for count in tile_counts.values()):
            return False
            
        # Pair must not be dragons or winds
        pair_tile = next((t for t, c in tile_counts.items() if c == 2), None)
        if not pair_tile or pair_tile in self.honors:
            return False
            
        # Must have sequences
        sequences = self._find_sequences(tiles)
        return len(sequences) >= 4
        
    def _check_iipeikou(self, tiles: List[str]) -> bool:
        """
        Check for iipeikou (pure double sequence).
        Must have exactly two identical sequences in the same suit.
        """
        sequences = self._find_sequences(tiles)
        # Convert sequences to tuples for counting
        sequence_tuples = [tuple(sorted(seq)) for seq in sequences]
        # Count occurrences of each sequence
        from collections import Counter
        sequence_counts = Counter(sequence_tuples)
        return any(count >= 2 for count in sequence_counts.values())
        
    def _check_sanshoku(self, tiles: List[str]) -> bool:
        """
        Check for sanshoku (three color straight).
        Same numbers across all three suits.
        """
        sequences = self._find_sequences(tiles)
        for i in range(1, 8):  # Check each possible starting number
            # Look for same sequence number in all three suits
            man_seq = {f"{i}m", f"{i+1}m", f"{i+2}m"}
            pin_seq = {f"{i}p", f"{i+1}p", f"{i+2}p"}
            sou_seq = {f"{i}s", f"{i+1}s", f"{i+2}s"}
            
            if (man_seq in [set(seq) for seq in sequences] and
                pin_seq in [set(seq) for seq in sequences] and
                sou_seq in [set(seq) for seq in sequences]):
                return True
        return False
        
    def _check_ittsuu(self, tiles: List[str]) -> bool:
        """
        Check for ittsuu (pure straight).
        Complete 1-9 sequence in a single suit.
        """
        sequences = self._find_sequences(tiles)
        for suit in ['m', 'p', 's']:
            # Check for 1-2-3, 4-5-6, and 7-8-9 in the same suit
            seq1 = {f"1{suit}", f"2{suit}", f"3{suit}"}
            seq2 = {f"4{suit}", f"5{suit}", f"6{suit}"}
            seq3 = {f"7{suit}", f"8{suit}", f"9{suit}"}
            
            if (seq1 in [set(seq) for seq in sequences] and
                seq2 in [set(seq) for seq in sequences] and
                seq3 in [set(seq) for seq in sequences]):
                return True
        return False
        
    def _find_sequences(self, tiles: List[str]) -> List[List[str]]:
        """Find all sequences in the hand."""
        sequences = []
        tile_counts = self.parse_tiles(tiles)
        
        # Check each suit
        for suit in ['m', 'p', 's']:
            # Check each possible starting number (1-7)
            for i in range(1, 8):
                seq = [f"{i}{suit}", f"{i+1}{suit}", f"{i+2}{suit}"]
                # Verify all three tiles exist
                if all(t in tile_counts and tile_counts[t] > 0 for t in seq):
                    sequences.append(seq)
                    
        return sequences
        
    def _check_honitsu(self, tiles: List[str]) -> Optional[str]:
        """Check if hand is one suit plus honors."""
        # Count tiles in each suit
        man = sum(1 for t in tiles if t.endswith('m'))
        pin = sum(1 for t in tiles if t.endswith('p'))
        sou = sum(1 for t in tiles if t.endswith('s'))
        honors = sum(1 for t in tiles if t in self.honors)
        
        # Should have tiles in exactly one suit plus honors
        suited_counts = [c for c in [man, pin, sou] if c > 0]
        if len(suited_counts) == 1 and honors > 0:
            if man > 0: return 'm'
            if pin > 0: return 'p'
            if sou > 0: return 's'
        return None
        
    def _check_chinitsu(self, tiles: List[str]) -> bool:
        """Check if hand is pure one suit (no honors)."""
        # Count tiles in each suit
        man = sum(1 for t in tiles if t.endswith('m'))
        pin = sum(1 for t in tiles if t.endswith('p'))
        sou = sum(1 for t in tiles if t.endswith('s'))
        honors = sum(1 for t in tiles if t in self.honors)
        
        # Should have tiles in exactly one suit and no honors
        suited_counts = [c for c in [man, pin, sou] if c > 0]
        return len(suited_counts) == 1 and honors == 0
    
    def _check_tanyao(self, tiles: List[str]) -> bool:
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
        recognizer = TileRecognizer()
        # Detect tiles in the image
        detected_tiles = self.tile_detector.detect_tiles(image)
        tiles_with_confidence = recognizer.recognize_multiple_tiles(detected_tiles)
        tiles = [tile for tile, confidence in tiles_with_confidence if confidence > 0.7]
        
        # Count yaku
        yaku_list = self.count_yaku(tiles)
        
        # Calculate total han
        total_han = sum(han for _, han in yaku_list)
        
        # Calculate score
        score_info = self.calculate_score(tiles, total_han)
        
        return {
            "detected_tiles": tiles,
            "yaku": yaku_list,
            "total_han": total_han,
            "score": score_info
        } 