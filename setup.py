from setuptools import setup, find_packages

# Read long description from README if available
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "GMOO SDK - Global Multi-Objective Optimization Software Development Kit"

setup(
    name="gmoo_sdk",
    version="2.0.0",
    author="Matt Freeman, Jason Hopkins",
    author_email="contact@globalmoo.com",
    description="A Python SDK for global multi-objective optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/globalMOO/gmoo-wrapper-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "numpy>=1.20.0",  # More widely compatible version
        "matplotlib>=3.4.0",  # For visualization tools
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
        "examples": [
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
        ],
    },
    python_requires=">=3.10",  # Required for secure DLL loading on Windows
)