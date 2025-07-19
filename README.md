# Mahjong Scorer

A Computer Vision-based scorer for Riichi mahjong using OpenCV.

## Features

- Tile recognition using computer vision
- Automatic scoring calculation
- Real-time game analysis

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone <repository-url>
   cd mahjong-scorer
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:

   ```bash
   pip install -e .
   ```

3. **Test the installation**:
   ```bash
   python test_opencv.py
   ```

### Dependencies

- **opencv-python**: Core OpenCV functionality
- **opencv-contrib-python**: Additional OpenCV modules
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **Pillow**: Image processing
- **scikit-image**: Advanced image processing

## Usage

Run the test script to verify everything is working:

```bash
python test_opencv.py
```

## Project Structure

```
mahjong-scorer/
├── mahjong_scorer/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── tile_detector.py     # OpenCV-based tile detection
│   └── scorer.py            # Mahjong scoring logic
├── config.py                # Configuration settings
├── example_usage.py         # Usage examples
├── test_opencv.py           # OpenCV installation test
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## Development

This project is set up for computer vision-based mahjong tile recognition. The test script includes functions that will be useful for:

- Tile detection using contour analysis
- Tile recognition using template matching
- Color analysis for different tile types
- Image preprocessing for better recognition

### Key Components

1. **TileDetector**: Uses OpenCV for image processing and tile detection
2. **MahjongScorer**: Handles mahjong rules and scoring calculations
3. **Configuration**: Centralized settings for easy customization

## License

MIT License - see LICENSE file for details.
