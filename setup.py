# setup.py — makes src/ importable as a package from anywhere in the project.
from setuptools import setup, find_packages

setup(
    name="llm-instruction-dilution",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
