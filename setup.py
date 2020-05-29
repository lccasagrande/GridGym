from os import path

from setuptools import find_packages, setup


def read_requirements_file(filename):
    file = '%s/%s' % (path.dirname(path.realpath(__file__)), filename)
    with open(file) as f:
        return [line.strip() for line in f]


with open("gridgym/__version__.py") as version_file:
    exec(version_file.read())


install_requires = read_requirements_file('requirements.txt')
tests_requires = read_requirements_file('requirements-dev.txt')
dev_requires = read_requirements_file('requirements-dev.txt')
setup_requires = ['pytest-runner']


setup(
    name='GridGym',
    version=__version__,
    license='MIT',
    author='lccasagrande',
    author_email='lcamelocasagrande@gmail.com',
    url='https://github.com/lccasagrande/gridgym',
    description="An OpenAI Gym environment for resource and job management problems.",
    python_requires='>=3.8',
    install_requires=install_requires,
    tests_require=tests_requires,
    setup_requires=setup_requires,
    extras_require={
        'dev': dev_requires,
    },
    packages=find_packages(),
    package_dir={'gridgym': 'gridgym'},
)
