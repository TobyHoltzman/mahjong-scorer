import unittest
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


from mahjong_scorer.scorer import MahjongScorer

class TestMahjongScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = MahjongScorer()

    def test_standard_winning_hands(self):
        """Test standard winning hands (4 groups + pair)"""
        # Three sequences, one triplet, one pair
        hand1 = [
            '1m', '2m', '3m',  # sequence
            '4p', '5p', '6p',  # sequence
            '7s', '8s', '9s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand1))

        # Two sequences, two triplets, one pair
        hand2 = [
            '1m', '2m', '3m',  # sequence
            '4p', '5p', '6p',  # sequence
            '7s', '7s', '7s',  # triplet
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand2))

    def test_hands_with_kan(self):
        """Test hands containing kan (four of a kind)"""
        # One kan, one triplet, one sequence, one pair
        hand1 = [
            '1m', '1m', '1m', '1m',  # kan
            '2p', '2p', '2p',  # triplet
            '3s', '4s', '5s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand1))

        # Two kan, one sequence, one pair
        hand2 = [
            '1m', '1m', '1m', '1m',  # kan
            '2p', '2p', '2p', '2p',  # kan
            '3s', '4s', '5s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand2))

        # Mixed sequences and kans
        hand3 = [
            '1m', '2m', '3m',  # sequence
            '4p', '4p', '4p', '4p',  # kan
            '7s', '8s', '9s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand3))

    def test_seven_pairs(self):
        """Test seven pairs pattern (chiitoitsu)"""
        hand = [
            '1m', '1m',  # pair 1
            '2p', '2p',  # pair 2
            '3s', '3s',  # pair 3
            'east', 'east',  # pair 4
            'red', 'red',  # pair 5
            'west', 'west',  # pair 6
            'north', 'north'  # pair 7
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand))

    def test_thirteen_orphans(self):
        """Test thirteen orphans pattern (kokushi musou)"""
        hand = [
            '1m', '9m',  # man terminals
            '1p', '9p',  # pin terminals
            '1s', '9s',  # sou terminals
            'east', 'south', 'west', 'north',  # winds
            'red', 'green', 'white',  # dragons
            'east'  # duplicate terminal/honor
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand))

    def test_invalid_hands(self):
        """Test various invalid hands"""
        # Wrong number of tiles
        hand1 = ['1m', '1m', '1m', '2m', '2m', '2m', '3m', '3m', '3m', '4m', '4m', '4m', '5m']
        self.assertFalse(self.scorer.is_valid_hand(hand1))

        # Too many of same tile
        hand2 = ['1m', '1m', '1m', '1m', '1m', '2m', '2m', '2m', '3m', '3m', '3m', '4m', '4m', '4m']
        self.assertFalse(self.scorer.is_valid_hand(hand2))

        # Invalid sequence
        hand3 = [
            '1m', '2m', '4m',  # broken sequence
            '4p', '5p', '6p',
            '7s', '8s', '9s',
            'east', 'east', 'east',
            'red', 'red'
        ]
        self.assertFalse(self.scorer.is_valid_hand(hand3))

        # Almost seven pairs but one missing
        hand4 = [
            '1m', '1m', '2p', '2p', '3s', '3s',
            'east', 'east', 'red', 'red', 'west',
            'west', 'north', 'south'
        ]
        self.assertFalse(self.scorer.is_valid_hand(hand4))

        # Almost thirteen orphans but missing one
        hand5 = [
            '1m', '9m', '1p', '9p', '1s', '9s',
            'east', 'south', 'west', 'north',
            'red', 'green', 'white', '5m'  # 5m instead of duplicate terminal/honor
        ]
        self.assertFalse(self.scorer.is_valid_hand(hand5))

        # Too few groups
        hand6 = [
            '1m', '1m', '1m', '1m',
            '4p', '4p', '4p', '4p',
            'east', 'east', 'east', 'east',
            'red', 'red'
        ]
        self.assertFalse(self.scorer.is_valid_hand(hand6))

        # Too many groups
        hand7 = [
            '1m', '2m', '3m',
            '4p', '5p', '6p',
            '7s', '8s', '9s',
            'east', 'east', 'east',
            'red', 'red', 'red', 'red',
            'south', 'south'
        ]
        self.assertFalse(self.scorer.is_valid_hand(hand7))

    def test_edge_cases(self):
        """Test edge cases and special situations"""
        # All terminals sequence-based hand
        hand1 = [
            '1m', '2m', '3m',
            '7m', '8m', '9m',
            '1p', '2p', '3p',
            '7p', '8p', '9p',
            '1s', '1s'
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand1))

        # All honors hand
        hand2 = [
            'east', 'east', 'east',
            'south', 'south', 'south',
            'west', 'west', 'west',
            'north', 'north', 'north',
            'red', 'red'
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand2))

        # Mixed sequences and kans
        hand3 = [
            '1m', '2m', '3m',  # sequence
            '4p', '4p', '4p', '4p',  # kan
            '7s', '8s', '9s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        self.assertTrue(self.scorer.is_valid_hand(hand3))

if __name__ == '__main__':
    unittest.main() 