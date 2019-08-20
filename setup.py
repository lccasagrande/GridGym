from setuptools import setup
from shutil import which
import sys


if which('batsim') is None:
    raise ImportError("(HINT: you need to install Batsim. Check the setup instructions here: https://batsim.readthedocs.io/en/latest/.)")

setup(name='GridGym',
      author='lccasagrande',
      version='0.0.1',
      python_requires='>=3.6',
      install_requires=[
              'gym',
              'numpy',
              'pandas',
              'zmq',
              'sortedcontainers',
              'evalys',
              'procset',
              'seaborn',
              'matplotlib'
      ])
