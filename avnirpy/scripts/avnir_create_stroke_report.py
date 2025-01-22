#!/usr/bin/env python3

"""
This script generates a stroke report in PDF format.

The script takes in three input files: a label image, a volume image, and a volumetry file.
It then generates a stroke report in PDF format, including patient information if provided.
"""

import argparse
from datetime import datetime
import json

from avnirpy.io.utils import (
    add_overwrite_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    add_version_arg,
)
from avnirpy.reporting.report import StrokeReport
from avnirpy.reporting.screenshot import (
    screenshot_mosaic_blend,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_labels", help="Path to the .nii.gz label image.")
    parser.add_argument("input_volume", help="Path to the .nii.gz volume image.")
    parser.add_argument("input_volumetry", help="Path to the .json volumetry file.")
    parser.add_argument("output_report", help="Path to the .pdf stroke report file.")

    parser.add_argument(
        "--patient_name",
        help="Patient name. Write the name between quotes.",
        default="Not available",
    )
    parser.add_argument("--patient_id", help="Patient ID.", default="Not available")

    add_overwrite_arg(parser)
    add_version_arg(parser)
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(
        parser, [args.input_labels, args.input_volume, args.input_volumetry]
    )
    assert_outputs_exist(parser, args, args.output_report)

    with open(args.input_volumetry, "r") as file:
        volumetry_data = json.load(file)

    report = StrokeReport(
        args.patient_name, args.patient_id, datetime.now().strftime("%d-%m-%Y")
    )

    screenshot_path = screenshot_mosaic_blend(
        args.input_volume,
        args.input_labels,
        min_val=0,
        max_val=140,
        nb_columns=5,
        offset_percent=0.3,
        blend_val=0.3,
        output_prefix="labels",
        directory=report.temp_dir,
    )
    report.render(volumetry_data, screenshot_path)
    report.to_pdf(args.output_report)
