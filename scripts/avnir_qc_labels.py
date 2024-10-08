#!/usr/bin/env python3

"""
Quality control of labels in NRRD format from 3DSlicer.
"""

import argparse
import logging
import json

import yaml

from avnirpy.io.image import load_nrrd, get_labels_from_nrrd_header
from avnirpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    check_segment_extent,
    check_images_space,
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
    parser.add_argument("labels", help="Path to the .nrrd label image.")
    parser.add_argument("volume", help="Path to the .nrrd volume image.")
    parser.add_argument("config", help="Path to the .yaml config file use in 3DSlicer.")
    parser.add_argument(
        "output", help="Path to the .json file containing the QC report."
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.labels, args.volume])
    assert_outputs_exist(parser, args, args.output)

    _, _, label_nrrdhearder, _ = load_nrrd(args.labels)
    _, _, volume_nrrdhearder, _ = load_nrrd(args.volume)
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    qc_extent = check_segment_extent(label_nrrdhearder)
    qc_space = check_images_space(volume_nrrdhearder, label_nrrdhearder)

    qc_labels = True
    labels_in_file = get_labels_from_nrrd_header(label_nrrdhearder)
    labels_in_config = {
        int(label["value"]): label["name"] for label in config["labels"]
    }
    for label, name in labels_in_file.items():
        if label not in labels_in_config:
            qc_labels = False
            logging.warning(
                f"Label {labels_in_file[label]} not found in the config file."
            )
        else:
            if name != labels_in_config[label]:
                qc_labels = False
                logging.warning(
                    f"Label {labels_in_file[label]} has a different name in the config file."
                )

    qc_report = {
        "global_qc": qc_extent and qc_space and qc_labels,
        "extent_qc": qc_extent,
        "space_qc": qc_space,
        "labels_qc": qc_labels,
    }
    with open(args.output, "w") as file:
        json.dump(qc_report, file)


if __name__ == "__main__":
    main()
