#!/usr/bin/env python3

"""
Quality control of labels in NRRD format from 3DSlicer.

If --output_json is provided, a json file will be created with the QC report and errors will be
printed in the console. Otherwise, the QC report will be printed in the console and errors will
stop the script.

Change the datatype of the labels and volume files for uint8 and int16 respectively.
"""

import argparse
import logging
import json

import yaml
import numpy as np

from avnirpy.io.image import load_nrrd, get_labels_from_nrrd_header, write_nrrd
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
    parser.add_argument("input_labels", help="Path to the .nrrd label image.")
    parser.add_argument("input_volume", help="Path to the .nrrd volume image.")
    parser.add_argument("config", help="Path to the .yaml config file use in 3DSlicer.")
    parser.add_argument(
        "output_labels", help="Path to the .nrrd label image corrected."
    )
    parser.add_argument(
        "output_volume", help="Path to the .nrrd volume image corrected."
    )
    parser.add_argument(
        "--output_json", help="Path to the .json file containing the QC report."
    )

    add_overwrite_arg(parser)
    add_verbose_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    log_func = logging.warning if args.output_json else parser.error

    assert_inputs_exist(parser, [args.input_labels, args.input_volume])
    assert_outputs_exist(
        parser, args, [args.output_labels, args.output_volume], args.output_json
    )

    label_data, _, label_nrrdhearder, label_affine = load_nrrd(args.input_labels)
    volume_data, _, volume_nrrdhearder, volume_affine = load_nrrd(args.input_volume)
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Check if the labels and volume are consistent
    qc_extent = check_segment_extent(label_nrrdhearder)
    if not qc_extent:
        log_func("The extent of each label are different.")
    qc_space = check_images_space(volume_nrrdhearder, label_nrrdhearder)
    if not qc_space:
        log_func("The space of the labels and volume images is different.")

    qc_labels = True
    labels_in_file, segment_match = get_labels_from_nrrd_header(label_nrrdhearder)
    labels_in_config = {
        label["name"]: int(label["value"]) for label in config["labels"]
    }
    qc_nb_labels = not (len(labels_in_file.keys()) > len(labels_in_config.keys()))
    if not qc_nb_labels:
        log_func(
            "The number of labels in the labels image file is greater than the number"
            "of labels in the config file."
        )

    # Check if all labels in the file are in the config file
    print(labels_in_file, labels_in_config)
    for name in labels_in_file.keys():
        if name not in labels_in_config:
            qc_labels = False
            log_func(f"Label {name} not found in the config file.")

    # Replace the labels in the file by the labels in the config file
    for name, label in labels_in_config.items():
        if name in labels_in_file and label != labels_in_file[name]:
            label_data[label_data == labels_in_file[name]] = label
            label_nrrdhearder[segment_match[label] + "_LabelValue"] = label
            log_func(
                f"Label {labels_in_file[label]} has a different name in the config file."
                "NRRD file modified."
            )

    # Save the corrected images
    write_nrrd(
        args.output_labels, label_data.astype(np.uint8), label_affine, label_nrrdhearder
    )
    write_nrrd(
        args.output_volume,
        volume_data.astype(np.int16),
        volume_affine,
        volume_nrrdhearder,
    )

    # Save the qc report
    if args.output_json:
        with open(args.output_json, "w") as file:
            json.dump(
                {
                    "global_qc": qc_extent and qc_space and qc_labels and qc_nb_labels,
                    "extent_qc": qc_extent,
                    "space_qc": qc_space,
                    "labels_qc": qc_labels,
                    "nb_labels_qc": qc_nb_labels,
                },
                file,
            )
    else:
        print(
            f"QC report: global={qc_extent and qc_space and qc_labels}, extent={qc_extent}, "
            f"space={qc_space}, labels={qc_labels}, qc_nb_labels={qc_nb_labels}"
        )


if __name__ == "__main__":
    main()
