from os import path

from setuptools import find_packages, setup


def read_requirements_file(filename):
    file = '%s/%s' % (path.dirname(path.realpath(__file__)), filename)
    with open(file) as f:
        return [line.strip() for line in f]


with open("gridgym/__version__.py") as version_file:
    exec(version_file.read())

with open("README.rst") as readme_file:
    long_description = readme_file.read()

install_requires = read_requirements_file('requirements.txt')


setup(
    name='gridgym',
    version=__version__,
    license='MIT',
    author='lccasagrande',
    author_email='lcamelocasagrande@gmail.com',
    url='https://github.com/lccasagrande/gridgym',
    download_url=f'https://github.com/lccasagrande/gridgym/archive/v{__version__}.tar.gz',
    description="An OpenAI Gym environment for resource and job management systems.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    python_requires='>=3.8',
    install_requires=install_requires,
    packages=find_packages(),
    package_dir={'gridgym': 'gridgym'},
    keywords=["Reinforcement Learning", "Scheduler",
              "Resource and Job Management", "Cluster"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: System :: Clustering",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
    ],
)
