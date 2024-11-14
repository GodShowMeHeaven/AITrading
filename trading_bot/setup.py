from setuptools import setup, find_packages

setup(
    name="trading_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ccxt>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
        "aiohttp>=3.8.5",
        "ta>=0.10.2",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.0.0",
        "seaborn>=0.12.0",
    ],
    python_requires=">=3.8",
)
