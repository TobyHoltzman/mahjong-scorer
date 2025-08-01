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
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities for PyTorch
- **tqdm**: Progress bars for training

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

# Test CNN-based recognition
python3 tests/test_cnn_recognition.py

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
│   ├── tile_recognition.py  # Tile recognition (CNN + template matching)
│   ├── train_cnn.py         # CNN training script
│   └── scorer.py            # Mahjong scoring logic
├── templates/               # Tile template images
├── training_data/           # Training data for CNN
├── models/                  # Trained CNN models
├── test_images/             # Generated test images (gitignored)
├── tests/                   # Test files and examples
│   ├── test_opencv.py       # OpenCV installation test
│   ├── test_cnn_recognition.py  # CNN recognition test
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

This project is set up for computer vision-based mahjong tile recognition using modern CNN-based recognition:

- Tile detection using contour analysis
- **Tile recognition using CNN with PyTorch**
- Color analysis for different tile types
- Image preprocessing for better recognition
- Deep learning model training and inference

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

3. **TileRecognizer**: Advanced tile recognition using CNN

   - CNN-based recognition with PyTorch
   - Pre-trained model support
   - High accuracy recognition
   - Confidence scoring
   - Model training capabilities

4. **MahjongTileCNN**: Custom CNN architecture for tile recognition

   - 4-layer convolutional network
   - Batch normalization and dropout
   - Optimized for 64x64 tile images
   - Support for 34 tile classes

5. **MahjongScorer**: Handles mahjong rules and scoring calculations

   - Yaku detection and scoring
   - Hand validation
   - Score calculation

6. **Configuration**: Centralized settings for easy customization

### Modular Design Benefits

- **Separation of Concerns**: Detection and recognition are independent
- **Easy Testing**: Can test each component separately
- **Flexible Recognition**: Can swap recognition methods without changing detection
- **Parameter Tuning**: Independent control over detection and recognition parameters
- **Maintainability**: Cleaner, more modular code structure

## Training the CNN Model

### Quick Training

To train the CNN model with synthetic data:

```bash
python3 -m mahjong_scorer.train_cnn
```

This will:

- Generate synthetic training data
- Train the model for 30 epochs
- Save the model to `models/mahjong_cnn.pth`
- Display training progress and results

### Custom Training

For custom training with your own data:

```python
from mahjong_scorer.train_cnn import train_cnn_model

# Train with custom parameters
model, class_names = train_cnn_model(
    data_dir="your_training_data",
    model_save_path="models/custom_model.pth",
    num_epochs=50,
    batch_size=64,
    learning_rate=0.001
)
```

### Training Data Structure

Organize your training data as follows:

```
training_data/
├── 1m/          # Man tiles
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── 2m/
│   └── ...
├── 1p/          # Pin tiles
│   └── ...
├── 1s/          # Sou tiles
│   └── ...
├── east/        # Honor tiles
│   └── ...
└── ...
```

### Model Architecture

The CNN uses a 4-layer architecture:

- **Input**: 64x64 RGB images
- **Conv1**: 32 filters, 3x3 kernel
- **Conv2**: 64 filters, 3x3 kernel
- **Conv3**: 128 filters, 3x3 kernel
- **Conv4**: 256 filters, 3x3 kernel
- **FC1**: 512 neurons
- **FC2**: 256 neurons
- **Output**: 34 classes (all mahjong tiles)

## License

MIT License - see LICENSE file for details.
