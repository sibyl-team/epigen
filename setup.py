from setuptools import setup, find_packages


setup(
    name="Sibyl-EpiGen",
    version="0.1",
    author="sibyl-team",
    packages=find_packages(),
    description="A package to generate synthetic epidemies on graphs",
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "networkx"
    ]
)
