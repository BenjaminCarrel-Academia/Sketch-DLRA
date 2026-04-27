import setuptools

with open("README.md", "r") as descr:
    long_description = descr.read()

setuptools.setup(
    name="sketch-dlra",
    version="1.0",
    author="Benjamin Carrel",
    author_email="benjamin.carrel@outlook.com",
    url="https://github.com/BenjaminCarrel/sketch-dlra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "pip",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
    ],
)
