"""Setup script for IRH Suite v9.2."""

from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="irh-suite",
    version="9.2.0",
    author="IRH Development Team",
    author_email="irh@example.com",
    description="Intrinsic Resonance Holography Suite - Computational Engine for Discrete Quantum Spacetime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dragonspider1991/Intrinsic-Resonance-Holography-",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "networkx>=3.1",
        "sympy>=1.12",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "gudhi>=3.8.0",
            "einsteinpy>=0.4.0",
            "astropy>=5.3",
            "scikit-learn>=1.3.0",
            "pulp>=2.7.0",
            "cvxpy>=1.4.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "sphinx>=7.2.0",
        ],
    },
)
