from setuptools import setup, find_packages

setup(
    name="investing",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "prophet",
        "plotly",
        "requests",
    ],
    description="Symphony Trading System for algorithmic trading",
    author="Asabaal",
    author_email="asabaal@example.com",
    url="https://github.com/asabaal/investing",
)
