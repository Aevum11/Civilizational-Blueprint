"""
Exception Theory - Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read the long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
version = {}
with open("exception_theory/__init__.py", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            exec(line, version)
            break

setup(
    name="exception-theory",
    version=version.get("__version__", "3.0.0"),
    author="M.J.M. (Exception Theory), ET Development Team (Implementation)",
    author_email="",  # Add email if desired
    description="A comprehensive mathematical framework for understanding reality through P, D, and T primitives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",  # Add repository URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        # No external dependencies - pure Python!
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    keywords="mathematics physics ontology theory manifold quantum consciousness distributed",
    project_urls={
        "Documentation": "",  # Add docs URL
        "Source": "",  # Add source URL
        "Bug Reports": "",  # Add issue tracker URL
    },
)
