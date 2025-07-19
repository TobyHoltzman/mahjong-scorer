# Mahjong Scorer Tests

This directory contains all test files for the mahjong scorer project.

## Test Files

### `test_opencv.py`

- **Purpose**: Verifies OpenCV installation and basic functionality
- **Tests**: OpenCV version, basic operations, mahjong-related functions
- **Usage**: `python test_opencv.py`

### `test_separation_no_templates.py`

- **Purpose**: Demonstrates separation of tile detection and recognition
- **Tests**: Independent detection, independent recognition, combined workflow
- **Usage**: `python test_separation_no_templates.py`
- **Note**: Uses mock templates, doesn't require actual template files

### `example_usage.py`

- **Purpose**: Example usage of the mahjong scorer system
- **Tests**: Manual tile analysis, image analysis, tile detection
- **Usage**: `python example_usage.py`

### `test_separation.py`

- **Purpose**: Tests separation with actual template files
- **Tests**: Template loading, recognition accuracy
- **Usage**: `python test_separation.py`
- **Note**: Requires template files in `../templates/` directory

## Running Tests

### Run All Tests

```bash
python run_all_tests.py
```

### Run Individual Tests

```bash
# From the tests directory
python test_opencv.py
python test_separation_no_templates.py
python example_usage.py

# From the project root
python tests/test_opencv.py
python tests/test_separation_no_templates.py
python tests/example_usage.py
```

## Test Categories

### **Installation Tests**

- `test_opencv.py` - Verifies OpenCV setup

### **Architecture Tests**

- `test_separation_no_templates.py` - Tests modular design
- `test_separation.py` - Tests with real templates

### **Usage Examples**

- `example_usage.py` - Demonstrates system usage

## Expected Results

### `test_opencv.py`

- ✅ OpenCV version displayed
- ✅ Basic operations working
- ✅ Mahjong functions available

### `test_separation_no_templates.py`

- ✅ Detection works independently
- ✅ Recognition works independently
- ✅ Combined workflow functional
- ✅ Parameter tuning demonstrated

### `example_usage.py`

- ✅ Manual tile analysis
- ✅ Image creation and analysis
- ✅ Tile detection demonstration

## Troubleshooting

### Import Errors

If you get import errors when running tests from the `tests/` directory:

```bash
# Run from project root instead
cd ..
python tests/test_name.py
```

### Template Errors

If template-related tests fail:

- Ensure `../templates/` directory exists
- Check that template files are present
- Use `test_separation_no_templates.py` for testing without templates

### OpenCV Errors

If OpenCV tests fail:

- Verify OpenCV installation: `pip install opencv-python`
- Check Python environment
- Ensure all dependencies are installed
