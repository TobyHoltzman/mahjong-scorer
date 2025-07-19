#!/usr/bin/env python3
"""
Test runner for all mahjong scorer tests.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_test(test_file: str) -> bool:
    """
    Run a single test file.
    
    Args:
        test_file: Path to the test file
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        # Add parent directory to path so tests can import mahjong_scorer
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))
        
        # Run the test with proper environment
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONPATH'] = str(parent_dir)
        
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, cwd=parent_dir,
                              env=env)
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"\n{status}: {test_file}")
        
        return success
        
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False


def main():
    """Run all tests in the tests directory."""
    print("Mahjong Scorer - Test Suite")
    print("=" * 40)
    
    # Get the tests directory
    tests_dir = Path(__file__).parent
    
    # Find all test files
    test_files = [
        "test_opencv.py",
        "test_separation_no_templates.py", 
        "example_usage.py"
    ]
    
    # Filter to only include files that exist
    existing_tests = []
    for test_file in test_files:
        test_path = tests_dir / test_file
        if test_path.exists():
            existing_tests.append(test_file)
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    if not existing_tests:
        print("No test files found!")
        return
    
    print(f"Found {len(existing_tests)} test files:")
    for test_file in existing_tests:
        print(f"  - {test_file}")
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_file in existing_tests:
        test_path = tests_dir / test_file
        if run_test(str(test_path)):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(existing_tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 