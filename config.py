"""
Configuration settings for the Mahjong Scorer.
"""

# OpenCV settings
CV_SETTINGS = {
    "min_contour_area": 1000,
    "max_contour_area": 50000,
    "min_aspect_ratio": 0.5,
    "max_aspect_ratio": 2.0,
    "gaussian_blur_kernel": (5, 5),
    "adaptive_threshold_block_size": 11,
    "adaptive_threshold_c": 2,
    "template_matching_threshold": 0.7,
}

# Mahjong game settings
GAME_SETTINGS = {
    "hand_size": 13,
    "min_han_for_win": 1,
    "default_fu": 30,
    "dealer_bonus_multiplier": 1.5,
}

# Scoring settings
SCORING_SETTINGS = {
    "mangan_threshold": 5,  # han
    "haneman_threshold": 6,  # han
    "baiman_threshold": 8,   # han
    "sanbaiman_threshold": 11,  # han
    "yakuman_threshold": 13,  # han
}

# File paths
PATHS = {
    "templates_dir": "templates/",
    "output_dir": "output/",
    "test_images_dir": "test_images/",
}

# Tile recognition settings
RECOGNITION_SETTINGS = {
    "enable_color_detection": True,
    "enable_template_matching": True,
    "enable_ml_recognition": False,  # Future feature
    "confidence_threshold": 0.8,
} 