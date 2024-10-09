#!/usr/bin/env python3

"""
Print the header of a nrrd or nifti image.
"""

import argparse
import json
import os

import numpy as np

from avnirpy.io.image import load_nrrd, load_nifti
from avnirpy.io.utils import assert_inputs_exist


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def has_nii_gz_extension(filename):
    base, ext = os.path.splitext(filename)
    return ext == ".gz" and os.path.splitext(base)[1] == ".nii"


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Path to the .nrrd or .nii.gz image.")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    if os.path.splitext(os.path.basename(args.input))[1] == ".nrrd":
        _, _, hdr, _ = load_nrrd(args.input)
        print(json.dumps(hdr, sort_keys=True, indent=4, cls=NumpyEncoder))
    elif has_nii_gz_extension(args.input):
        _, hdr, _ = load_nifti(args.input)
        print(hdr)
    else:
        print("File extension not supported. Please use .nrrd or .nii.gz.")


if __name__ == "__main__":
    main()
