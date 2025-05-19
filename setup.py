from setuptools import setup, find_packages

setup(
    name="datacenter_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "plotly",
        "dash",
        "seaborn",
        "matplotlib"
    ],
    python_requires=">=3.8",
) 