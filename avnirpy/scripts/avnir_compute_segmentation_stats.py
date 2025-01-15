#!/usr/bin/env python3

"""
Save images information in a .csv file.
"""

import argparse
import logging
import os

from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
import pandas as pd

from avnirpy.io.image import load_nifti
from avnirpy.io.utils import (
    assert_inputs_exist,
    assert_outputs_exist,
    add_version_arg,
    add_overwrite_arg,
    add_verbose_arg,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "ground_truth", help="Directory containing the ground truth segmentations."
    )
    parser.add_argument(
        "predictions",
        help="Directory containing the predicted segmentations.",
    )
    parser.add_argument("output", help="Path to the .csv statistical report.")

    add_overwrite_arg(parser)
    add_verbose_arg(parser)
    add_version_arg(parser)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(
        parser, [args.ground_truth, args.predictions], is_directory=True
    )
    assert_outputs_exist(parser, args, args.output)

    if not args.output.endswith(".csv"):
        args.output = args.output + ".csv"

    data = []
    for i in os.listdir(args.predictions):
        if not os.path.exists(os.path.join(args.ground_truth, i)):
            logging.warning(f"Segmentation {i} not found in both directories.")
            continue

        prediction, _, _ = load_nifti(os.path.join(args.predictions, i))
        reference, _, _ = load_nifti(os.path.join(args.ground_truth, i))

        bpm = BPM(prediction, reference, measures=None)
        dict_seg = bpm.to_dict_meas()
        dict_seg["image"] = i
        data.append(dict_seg)

    df = pd.DataFrame(data)
    image = df["image"]
    df.drop(labels=["image"], axis=1, inplace=True)
    df.insert(0, "image", image)
    df.to_csv(args.output, index=False)
    df.describe().to_csv(args.output.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
