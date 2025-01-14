#!/usr/bin/env python3

"""
Save images information in a .csv file.
"""

import argparse
import json
import os
import csv

import numpy as np
import pandas as pd

from avnirpy.io.image import load_nrrd, load_nifti
from avnirpy.io.utils import (
    assert_inputs_exist,
    assert_outputs_exist,
    add_version_arg,
    add_overwrite_arg,
)


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
    parser.add_argument("input", nargs="+", help="Path to the .nrrd or .nii.gz image.")
    parser.add_argument("output", help="Path to the .csv report.")

    add_overwrite_arg(parser)
    add_version_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    if not args.output.endswith(".csv"):
        args.output = args.output + ".csv"

    data = []
    for image in args.input:
        if os.path.splitext(os.path.basename(image))[1] == ".nrrd":
            _, hdr, _, _ = load_nrrd(image)
        elif has_nii_gz_extension(image):
            _, hdr, _ = load_nifti(image)
        else:
            print(
                f"File extension not supported for {image}. Please use .nrrd or .nii.gz."
            )

        data.append(
            {
                "Image": image,
                "voxel_size x": hdr["pixdim"][1],
                "voxel_size y": hdr["pixdim"][2],
                "voxel_size z": hdr["pixdim"][3],
                "shape x": hdr["dim"][1],
                "shape y": hdr["dim"][2],
                "shape z": hdr["dim"][3],
            }
        )

    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    df.describe().to_csv(args.output.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
