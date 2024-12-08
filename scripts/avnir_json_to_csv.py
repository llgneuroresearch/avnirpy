#!/usr/bin/env python3

"""
Convert a JSON file to a CSV file using pandas.
"""

import argparse
import logging
import pandas as pd
import json
from avnirpy.io.utils import add_version_arg

from avnirpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_json", help="Path to the input .json file.")
    parser.add_argument("output_csv", help="Path to the output .csv file.")

    add_overwrite_arg(parser)
    add_version_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.input_json])
    assert_outputs_exist(parser, args, [args.output_csv])

    with open(args.input_json, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df = df[sorted(df.columns)]

    df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
