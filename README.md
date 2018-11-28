## Overview

This repository contains a grid-system simulator based on [**OpenAI Gym**](https://github.com/openai/gym) and [**Batsim**](https://github.com/oar-team/batsim)

## Requirements

You'll need **Python 3** and **Docker** to be able to run theses projects. 

If you do not have Python installed yet, it is recommended that you install the [Anaconda](https://www.anaconda.com/download/) distribution of Python, which has almost all packages required in these projects. 

You can also install Python 3 from [here](https://www.python.org/)

Docker can be found here [here](https://www.docker.com/)

## Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/lccasagrande/GridGym.git
cd GridGym
```
2. Get the stable version (temporary*).
```
git pull origin stable
```

3. Install required packages: 
```
pip install -e .
```
4. Run an example:
```	
python examples/sjf.py
```
If you have any questions or find a bug, please contact me!