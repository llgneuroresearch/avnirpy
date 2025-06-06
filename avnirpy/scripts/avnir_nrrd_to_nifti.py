#!/usr/bin/env python3

"""
Convert nrrd image to nifti image.
"""

import argparse

import nibabel as nib

from avnirpy.io.image import load_nrrd
from avnirpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    add_version_arg,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Path to the .nrrd image.")
    parser.add_argument("output", help="Path to the .nii.gz image.")

    add_overwrite_arg(parser)
    add_version_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    assert_outputs_exist(parser, args, args.output)

    data, niiheader, nrrdhearder, affine = load_nrrd(args.input)
    img_nifti = nib.nifti1.Nifti1Image(data, affine=affine, header=niiheader)
    nib.save(img_nifti, args.output)


if __name__ == "__main__":
    main()
