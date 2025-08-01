from setuptools import setup, find_packages

setup(
    name="mahjong-scorer",
    version="0.1.0",
    description="Scorer for Riichi mahjong using Computer Vision",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.9.0",
        "opencv-contrib-python>=4.9.0",
        "numpy>=1.26.0",
        "matplotlib>=3.8.0",
        "Pillow>=10.1.0",
        "scikit-image>=0.22.0",
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "tqdm>=4.66.0",
    ],
    python_requires=">=3.13.5",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.13",
    ],
) 