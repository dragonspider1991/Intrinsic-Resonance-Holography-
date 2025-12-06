"""
Setup configuration for Intrinsic Resonance Holography v11.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README_v11.md").read_text()

setup(
    name="intrinsic-resonance-holography",
    version="11.0.0",
    author="Brandon D. McCrary",
    author_email="brandon.mccrary@example.com",
    description="A unified theory deriving physics from information-theoretic first principles",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'matplotlib>=3.7.0',
        'networkx>=3.1',
        'scikit-learn>=1.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=8.0.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'ruff>=0.1.0',
            'mypy>=1.5.0',
        ],
        'quantum': [
            'qutip>=5.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'ipykernel>=6.25.0',
        ],
    },
)
