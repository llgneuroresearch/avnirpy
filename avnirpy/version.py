# -*- coding: utf-8 -*-

import glob
import os

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = 0
_version_extra = ""

# Construct full version string from these.
_ver = [_version_major, _version_minor, _version_micro]

if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "Avnirpy: A brain segmentation toolbox."
# Long description will go up on the pypi page
long_description = """
Avnirpy
========
Avnirpy is a library developed by Avnir Lab under the direction of Dr. Letourneau Guillon.
It includes a range of tools primarily designed for CT scans segmentation and analysis.

License
=======
``avnirpy`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2024--, Avnir Lab,
Centre de recherche du centre hospitalier de l'unversite de Montreal.
"""

NAME = "avnirpy"
MAINTAINER = "Guillaume Theaud"
MAINTAINER_EMAIL = "guillaume.theaud.chum@ssss.gouv.qc.ca"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/llgneuroresearch/avnirpy"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "The Onset developers"
AUTHOR_EMAIL = ""
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
SCRIPTS = filter(
    lambda s: not os.path.basename(s) == "__init__.py", glob.glob("scripts/*.py")
)
PYTHON_VERSION = ">=3.9,<3.12"
