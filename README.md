# Mahjong Scorer

A Computer Vision-based scorer for Riichi mahjong using OpenCV.

## Features

- Tile recognition using computer vision
- Automatic scoring calculation
- Real-time game analysis

## Setup

### Prerequisites

- Python 3.13.5 or higher
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

3. **Verify Python 3.13.5 setup**:

   ```bash
   python3 verify_python_version.py
   ```

4. **Test the installation**:
   ```bash
   python3 tests/test_opencv.py
   ```

### Dependencies

- **opencv-python**: Core OpenCV functionality
- **opencv-contrib-python**: Additional OpenCV modules
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **Pillow**: Image processing
- **scikit-image**: Advanced image processing

## Usage

### Quick Start

First, verify your Python 3.13.5 setup:

```bash
python3 verify_python_version.py
```

Then run the test script to verify everything is working:

```bash
python3 tests/test_opencv.py
```

### Run All Tests

```bash
python3 tests/run_all_tests.py
```

### Individual Tests

```bash
# Test OpenCV installation
python3 tests/test_opencv.py

# Test modular design (no templates required)
python3 tests/test_separation_no_templates.py

# Example usage
python3 tests/example_usage.py
```

## Project Structure

```
mahjong-scorer/
├── mahjong_scorer/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── tile_detector.py     # Tile detection (contour analysis)
│   ├── tile_recognition.py  # Tile recognition (template matching)
│   └── scorer.py            # Mahjong scoring logic
├── templates/               # Tile template images
├── test_images/             # Generated test images (gitignored)
├── tests/                   # Test files and examples
│   ├── test_opencv.py       # OpenCV installation test
│   ├── test_separation_no_templates.py  # Separation test
│   ├── example_usage.py     # Usage examples
│   ├── run_all_tests.py     # Test runner
│   └── README.md            # Test documentation
├── config.py                # Configuration settings
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

1. **TileDetector**: Handles tile detection using contour analysis

   - Image preprocessing
   - Contour detection and filtering
   - Tile region extraction
   - Configurable detection parameters

2. **TileRecognizer**: Handles tile recognition using template matching

   - Template loading and management
   - Template matching with confidence scores
   - Multiple recognition methods support
   - Configurable confidence thresholds

3. **MahjongScorer**: Handles mahjong rules and scoring calculations

   - Yaku detection and scoring
   - Hand validation
   - Score calculation

4. **Configuration**: Centralized settings for easy customization

### Modular Design Benefits

- **Separation of Concerns**: Detection and recognition are independent
- **Easy Testing**: Can test each component separately
- **Flexible Recognition**: Can swap recognition methods without changing detection
- **Parameter Tuning**: Independent control over detection and recognition parameters
- **Maintainability**: Cleaner, more modular code structure

## License

MIT License - see LICENSE file for details.
