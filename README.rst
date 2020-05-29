GridGym
========

An `OpenAI Gym <https://github.com/openai/gym>`_ environment for resource and job management problems which, use `Batsim <https://github.com/oar-team/batsim>`_ to simulate the Resource and Job Management System (RJMS) behavior.

Installation
------------


1. Make sure you have Batsim v3.1.0 installed and working. Otherwise, you must follow `Batsim installation <https://batsim.readthedocs.io/en/latest/installation.html>`_ instructions. Check the version of Batsim with:

.. code-block:: bash

    batsim --version

2: Install GridGym:

.. code-block:: bash

    git clone https://github.com/lccasagrande/GridGym.git
    cd GridGym
    pip install -e .
