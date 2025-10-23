from setuptools import setup, find_packages

setup(
    name="n2n",
    version="0.0.2",
    author="",
    author_email="",
    description="Noise2Noise 1D waveform denoising (TCN)",
    long_description="Noise2Noise 1D waveform denoising package",
    long_description_content_type="text/markdown",
    python_requires=">=3.10, <3.14",
    packages=find_packages(),
    include_package_data=True,
)


