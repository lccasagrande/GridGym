import os
from setuptools import setup
#from shutil import which
import sys


#if which('batsim') is None:
#    raise ImportError("(HINT: you need to install Batsim. Check the setup instructions here: https://batsim.readthedocs.io/en/latest/.)")

os.makedirs("/tmp/GridGym/workloads/", exist_ok=True)
os.makedirs("/tmp/GridGym/output/", exist_ok=True)

setup(name='GridGym',
      author='lccasagrande',
      version='0.0.1',
      python_requires='>=3.7',
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
