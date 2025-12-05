#!/usr/bin/env python
"""
Setup script for CNCG package.

This provides backward compatibility with older pip versions
and tools that don't support pyproject.toml.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cncg",
    version="14.0.0",
    description="Computational Non-Commutative Geometry: Spontaneous Emergence of Four Dimensions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Brandon D. McCrary",
    author_email="independent@researcher.org",
    url="https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-",
    project_urls={
        "Homepage": "https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-",
        "Repository": "https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-",
    },
    license="CC0 1.0 Universal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords=["non-commutative geometry", "spectral action", "quantum gravity", "theoretical physics"],
)
