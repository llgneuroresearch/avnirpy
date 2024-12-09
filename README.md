# Avnirpy
[![codecov](https://codecov.io/github/llgneuroresearch/avnirpy/graph/badge.svg?token=P97KLITHA0)](https://codecov.io/github/llgneuroresearch/avnirpy)
![Tests](https://github.com/llgneuroresearch/avnirpy/actions/workflows/test.yml/badge.svg?branch=main)
[![Docker](https://github.com/llgneuroresearch/avnirpy/actions/workflows/docker.yml/badge.svg?branch=main)](https://hub.docker.com/r/avnirlab/avnirpy/tags)

Avnirpy is a library developed by Avnir Lab under the direction of Dr. Letourneau Guillon.
It includes a range of tools primarily designed for CT scans segmentation and analysis.

## Installation

### Prerequisites

- Python >=3.9 and <3.12
- Git

### Steps

1. **Clone the Repository**

	```sh
	git clone git@github.com:llgneuroresearch/avnirpy.git
	cd avnirpy
	```

2. **Install Avnirpy**

	```sh
	pip install -e .
	```

4. **Run Tests**

	Ensure that the installation is successful by running the tests:

	```sh
	pytest
	```

If all tests pass, you have successfully set up the development environment for Avnirpy.
