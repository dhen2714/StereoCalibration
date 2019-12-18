from setuptools import setup, Extension, find_packages
from stereocal import __version__


requirements = [
    'setuptools>=18.0',
    'opencv-python',
    'opencv-contrib-python',
    'pandas',
    'sklearn',
    'numpy',
    'matplotlib'
]


setup(
    name='stereocal',
    version=__version__,
    install_requires=requirements,
    packages=find_packages(),
    description='Tools for stereo calibration.',
    license='GNU Lesser General Public License v3 (LGPLv3)',
)