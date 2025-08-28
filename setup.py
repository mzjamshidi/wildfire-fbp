from setuptools import setup, find_packages

setup(
    name="wildfire-fbp",
    version="0.1.0",
    author="Maziar Jamshidi",
    packages=find_packages(exclude=("data",)),
    install_requires=[
        "numpy",
        "scikit-image",
        "rasterio",
    ]
)