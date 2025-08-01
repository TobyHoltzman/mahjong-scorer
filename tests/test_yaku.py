import unittest, sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from mahjong_scorer.scorer import MahjongScorer

class TestYakuCounting(unittest.TestCase):
    def setUp(self):
        self.scorer = MahjongScorer()

    def test_kokushi_musou(self):
        """Test thirteen orphans pattern"""
        hand = [
            '1m', '9m',  # man terminals
            '1p', '9p',  # pin terminals
            '1s', '9s',  # sou terminals
            'east', 'south', 'west', 'north',  # winds
            'red', 'green', 'white',  # dragons
            'east'  # duplicate terminal/honor
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertEqual(len(yaku), 1)
        self.assertEqual(yaku[0], ("Kokushi Musou", 13))

    def test_chiitoitsu(self):
        """Test seven pairs pattern"""
        # Seven pairs with tanyao
        hand = [
            '2m', '2m', '3p', '3p', '4s', '4s',
            '5m', '5m', '6p', '6p', '7s', '7s',
            '8m', '8m'
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertEqual(len(yaku), 2)
        self.assertIn(("Chiitoitsu", 2), yaku)
        self.assertIn(("Tanyao", 1), yaku)

    def test_honitsu(self):
        """Test one suit plus honors pattern"""
        hand = [
            '1m', '2m', '3m',  # sequence
            '4m', '5m', '6m',  # sequence
            '7m', '8m', '9m',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Honitsu", 3), yaku)

    def test_chinitsu(self):
        """Test pure one suit pattern"""
        hand = [
            '1m', '2m', '3m',  # sequence
            '4m', '5m', '6m',  # sequence
            '7m', '8m', '9m',  # sequence
            '2m', '2m', '2m',  # triplet
            '5m', '5m'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Chinitsu", 6), yaku)

    def test_multiple_kans(self):
        """Test hands with multiple kans"""
        # Three kans
        hand = [
            '1m', '1m', '1m', '1m',  # kan
            '2p', '2p', '2p', '2p',  # kan
            '3s', '3s', '3s', '3s',  # kan
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("San Kantsu", 2), yaku)
        self.assertIn(("Toitoi", 2), yaku)

        # Two kans
        hand = [
            '1m', '1m', '1m', '1m',  # kan
            '2p', '2p', '2p', '2p',  # kan
            '3s', '4s', '5s',  # sequence
            'east', 'east', 'east',  # triplet
            'red', 'red'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Ryan Kantsu", 2), yaku)

    def test_yakuhai(self):
        """Test valuable tiles pattern"""
        hand = [
            '1m', '2m', '3m',  # sequence
            'red', 'red', 'red',  # dragon triplet
            'green', 'green', 'green',  # dragon triplet
            'east', 'east', 'east',  # wind triplet
            'white', 'white'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Yakuhai", 3), yaku)  # 3 valuable triplets

    def test_tanyao(self):
        """Test all simples pattern"""
        hand = [
            '2m', '3m', '4m',  # sequence
            '5p', '6p', '7p',  # sequence
            '3s', '4s', '5s',  # sequence
            '7m', '7m', '7m',  # triplet
            '8p', '8p'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Tanyao", 1), yaku)

    def test_pinfu(self):
        """Test pinfu (no points hand)"""
        hand = [
            '2m', '3m', '4m',  # sequence
            '5p', '6p', '7p',  # sequence
            '3s', '4s', '5s',  # sequence
            '6s', '7s', '8s',  # sequence
            '2p', '2p'  # pair (not dragons/winds)
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Pinfu", 1), yaku)
        self.assertIn(("Tanyao", 1), yaku)  # This hand is also tanyao

    def test_iipeikou(self):
        """Test iipeikou (pure double sequence)"""
        hand = [
            '1m', '2m', '3m',  # sequence
            '1m', '2m', '3m',  # same sequence
            '1p', '2p', '3p',  # different sequence
            '2s', '3s', '4s',  # different sequence
            'red', 'red'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Iipeikou", 1), yaku)

    def test_sanshoku(self):
        """Test sanshoku (three color straight)"""
        hand = [
            '3m', '4m', '5m',  # sequence
            '3p', '4p', '5p',  # same sequence different suit
            '3s', '4s', '5s',  # same sequence different suit
            '1m', '1m', '1m',  # triplet
            'red', 'red'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Sanshoku Doujun", 2), yaku)

    def test_ittsuu(self):
        """Test ittsuu (pure straight)"""
        hand = [
            '1m', '2m', '3m',  # 1-2-3
            '4m', '5m', '6m',  # 4-5-6
            '7m', '8m', '9m',  # 7-8-9
            'red', 'red', 'red',  # triplet
            'east', 'east'  # pair
        ]
        yaku = self.scorer.count_yaku(hand)
        self.assertIn(("Ittsuu", 2), yaku)

    def test_invalid_yaku(self):
        """Test hands that look similar to yaku but aren't valid"""
        # Almost pinfu but has a triplet
        hand1 = [
            '2m', '3m', '4m',
            '5p', '6p', '7p',
            '3s', '3s', '3s',  # triplet invalidates pinfu
            '6s', '7s', '8s',
            '2p', '2p'
        ]
        yaku = self.scorer.count_yaku(hand1)
        self.assertNotIn(("Pinfu", 1), yaku)

        # Almost iipeikou but sequences in different suits
        hand2 = [
            '1m', '2m', '3m',
            '1p', '2p', '3p',  # same sequence but different suit
            '1s', '2s', '3s',
            '2s', '3s', '4s',
            'red', 'red'
        ]
        yaku = self.scorer.count_yaku(hand2)
        self.assertNotIn(("Iipeikou", 1), yaku)

        # Almost sanshoku but one number off
        hand3 = [
            '3m', '4m', '5m',
            '3p', '4p', '5p',
            '4s', '5s', '6s',  # different sequence
            '1m', '1m', '1m',
            'red', 'red'
        ]
        yaku = self.scorer.count_yaku(hand3)
        self.assertNotIn(("Sanshoku Doujun", 2), yaku)

        # Almost ittsuu but missing middle sequence
        hand4 = [
            '1m', '2m', '3m',
            '4p', '5p', '6p',  # wrong suit
            '7m', '8m', '9m',
            'red', 'red', 'red',
            'east', 'east'
        ]
        yaku = self.scorer.count_yaku(hand4)
        self.assertNotIn(("Ittsuu", 2), yaku)

if __name__ == '__main__':
    unittest.main() 