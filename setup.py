from setuptools import setup
import sys

CURRENT_PYTHON = float("{}.{}".format(sys.version_info.major, sys.version_info.minor))
REQUIRED_PYTHON = 3

if CURRENT_PYTHON < REQUIRED_PYTHON:
	sys.stderr.write("""
        ==========================
        Unsupported Python version
        ==========================

        This project requires Python {}, but you're trying to install it on Python {}.        
        """.format(REQUIRED_PYTHON, CURRENT_PYTHON))

setup(name='GridGym',
      python_requires='>={}'.format(REQUIRED_PYTHON),
      author='lccasagrande',
      version='0.0.1',
      install_requires=[
	      'gym',
	      'numpy',
	      'pandas',
	      'zmq',
	      'sortedcontainers',
	      'matplotlib'
      ])
