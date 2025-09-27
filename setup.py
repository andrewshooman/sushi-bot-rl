#!/usr/bin/env python3
"""
Setup script for Sushi Bot RL
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sushi-bot-rl",
    version="1.0.0",
    author="andrewshooman",
    author_email="sushibubba@gmail.com",
    description="A sophisticated Rocket League reinforcement learning bot",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/andrewshooman/sushi-bot-rl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sushi-bot-train=train:main",
            "sushi-bot-standardize=standardize_checkpoints:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.cfg", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/andrewshooman/sushi-bot-rl/issues",
        "Source": "https://github.com/andrewshooman/sushi-bot-rl",
        "Documentation": "https://github.com/andrewshooman/sushi-bot-rl/docs",
    },
)