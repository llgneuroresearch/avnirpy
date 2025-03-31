#!/usr/bin/env python3

"""
Script to compute segmentation statistics between ground truth and predicted segmentations.

This script compares segmentation results stored in two directories: one containing 
the ground truth segmentations and the other containing the predicted segmentations. 
It calculates various statistical measures for each segmentation pair using the 
`MetricsReloaded` (https://metrics-reloaded.dkfz.de/metric-library)
library and outputs the results in a CSV file. Optionally, it can 
also compute statistics for individual labels in a multi-label segmentation scenario.

Example:

    avnir_compute_segmentation_stats \\
    /path/to/ground_truth \\
    /path/to/predictions \\
    /path/to/output.csv \\
    --multilabel --verbose
"""

import argparse
import logging
import os

from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures as BPM
import pandas as pd
import numpy as np

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

    parser.add_argument(
        "--multilabel", action="store_true", help="Use multi-label mode."
    )

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
        dict_seg["label"] = "all"
        data.append(dict_seg)

        if args.multilabel:
            labels = np.unique(reference)
            for label in labels:
                if label == 0:
                    continue
                prediction_label = (prediction == label).astype(int)
                reference_label = (reference == label).astype(int)
                bpm = BPM(prediction_label, reference_label, measures=None)
                dict_seg = bpm.to_dict_meas()
                dict_seg["image"] = i
                dict_seg["label"] = label
                data.append(dict_seg)

    df = pd.DataFrame(data)
    df = df[
        ["image", "label"]
        + [col for col in df.columns if col not in ["image", "label"]]
    ]
    if args.multilabel:
        df_labels = df[df["label"] != "all"]
        df_labels.to_csv(
            args.output.replace(".csv", "_multilabel.csv"),
            index=False,
        )
        df_labels.describe().to_csv(
            args.output.replace(".csv", "_multilabel_summary.csv"), index=True
        )

        df_all = df[df["label"] == "all"]
        df_all.to_csv(args.output.replace(".csv", "_all.csv"), index=False)
        df_all.describe().to_csv(
            args.output.replace(".csv", "_all_summary.csv"), index=True
        )
    else:
        df.to_csv(args.output, index=False)
        df.describe().to_csv(args.output.replace(".csv", "_summary.csv"), index=True)


if __name__ == "__main__":
    main()
