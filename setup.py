from setuptools import setup, find_packages


setup(
    name="Sibyl-EpiGen",
    version="0.1",
    author="sibyl-team",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "networkx"
    ]
)
