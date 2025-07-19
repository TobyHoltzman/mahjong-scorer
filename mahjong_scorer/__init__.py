"""
Mahjong Scorer - Computer Vision-based Riichi Mahjong Scoring System
"""

__version__ = "0.1.0"
__author__ = "Alan Cheng, Toby Holtzman"

from .tile_detector import TileDetector
from .tile_recognition import TileRecognizer, MahjongTileCNN
from .scorer import MahjongScorer

__all__ = ["TileDetector", "TileRecognizer", "MahjongTileCNN", "MahjongScorer"] 