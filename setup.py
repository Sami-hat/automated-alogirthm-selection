# setup.py
from setuptools import setup, find_packages

setup(
    name="algorithm-selection-framework",
    version="1.0.0",
    description="A framework for Algorithm Selection using machine learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "optuna>=2.10.0",
        "jinja2>=3.0.0",
        "joblib>=1.0.0",
        "scipy>=1.7.0",
        "tqdm>=4.62.0",
    ],
    entry_points={
        "console_scripts": [
            "as-experiment=main:main",
        ],
    },
)