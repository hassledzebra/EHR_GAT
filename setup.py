#!/usr/bin/env python
"""
Setup script for EPI: Epilepsy Prediction using Heterogeneous Graph Attention Networks
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="epi-prediction",
    version="1.0.0",
    author="EPI Research Team",
    author_email="your.email@domain.com",
    description="Epilepsy Prediction using Heterogeneous Graph Attention Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/epi-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "flake8>=3.9.0",
            "black>=21.0.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch-scatter>=2.0.0",
            "torch-sparse>=0.6.0",
            "torch-cluster>=1.5.0",
            "torch-spline-conv>=1.2.0",
        ],
        "spark": [
            "pyspark>=3.0.0",
            "findspark>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "epi-train=train_hetero_gat_model:main",
            "epi-compare=run_full_model_comparison:main",
            "epi-dashboard=dashboard:serve",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    keywords="epilepsy, prediction, graph neural networks, medical AI, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/your-org/epi-prediction/issues",
        "Source": "https://github.com/your-org/epi-prediction",
        "Documentation": "https://docs.your-org.com/epi",
    },
)