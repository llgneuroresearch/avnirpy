#!/usr/bin/env python3

"""
This script generates a stroke report in PDF format.

The script takes in three input files: a label image, a volume image, and a volumetry file.
It then generates a stroke report in PDF format, including patient information if provided.
"""

import argparse
from datetime import datetime
import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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
from avnirpy.reporting.screenshot import colors


def compute_diff_perc(current_df, previous_df, column, label):
    previous = (
        previous_df.loc[previous_df["label_name"] == label, column].values[0]
        if label in previous_df["label_name"].values
        else None
    )
    current = current_df.loc[current_df["label_name"] == label, column].values[0]
    current_df.loc[current_df["label_name"] == label, f"last_{column}"] = previous
    current_df.loc[current_df["label_name"] == label, f"diff_{column}"] = (
        current - previous
    )
    current_df.loc[current_df["label_name"] == label, f"diff_perc_{column}"] = (
        (current - previous) / previous * 100
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
    parser.add_argument(
        "date_time", help="Date and time of the report. Format: YYYYMMDDHHMMSS"
    )
    parser.add_argument("output_report", help="Path to the .pdf stroke report file.")

    parser.add_argument(
        "--patient_name",
        help="Patient name. Write the name between quotes.",
        default="Not available",
    )
    parser.add_argument("--patient_id", help="Patient ID.", default="Not available")
    parser.add_argument("--previous_timepoint", help="Previous timepoint.")
    parser.add_argument(
        "--output_longitudinal", help="Path to the .json longitudinal data."
    )

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

    label_name = {1: "EDH", 2: "IPH", 3: "IVH", 4: "SAH", 5: "SDH"}
    current_df = pd.read_json(args.input_volumetry, precise_float=True)
    current_df.sort_values(by="volume", ascending=False, inplace=True)
    current_df = current_df.round(15)
    current_df["date"] = datetime.strptime(args.date_time, "%Y%m%d%H%M%S")
    current_df["label_name"] = current_df["label_id"].map(label_name)

    report = StrokeReport(
        args.patient_name, args.patient_id, datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    )

    timepoint_graphs = None
    if args.previous_timepoint:
        labels = current_df["label_name"].unique()
        timepoint_df = pd.read_json(args.previous_timepoint, precise_float=True)

        # Process timepoint data
        timepoint_df["label_name"] = timepoint_df["label_id"].map(label_name)

        # Get previous timepoint data
        previous_df = timepoint_df.loc[
            timepoint_df.groupby("label_name")["date"].idxmax()
        ]

        # Calculate differences
        for label in labels:
            compute_diff_perc(current_df, previous_df, "volume", label)
            if "volume_normalized" in current_df.columns:
                compute_diff_perc(current_df, previous_df, "volume_normalized", label)

        # Generate timepoint graphs
        all_timepoint_df = pd.concat([timepoint_df, current_df], ignore_index=True)
        timepoint_graphs = []
        for label in labels:
            label_data = all_timepoint_df[
                all_timepoint_df["label_name"] == label
            ].sort_values(by="date")
            plt.figure(figsize=(14, 9))
            sns.set_theme(style="dark")
            sns.set_context("talk", font_scale=2)
            sns.lineplot(
                data=label_data,
                x="date",
                y="volume",
                marker="o",
                linewidth=4.5,
                markersize=12,
                label=f"{label}",
                color=[*colors][int(label_data["label_id"].values[0])],
            )
            plt.title(
                f"{label} - Volume Evolution Over Time", fontsize=36, fontweight="bold"
            )
            plt.gca().xaxis.set_major_formatter(
                plt.matplotlib.dates.DateFormatter("%d-%m %H:%M")
            )
            plt.xlabel("Date Hour", fontsize=32, fontweight="bold")
            plt.ylabel("Volume (ml)", fontsize=32, fontweight="bold")
            plt.xticks(fontsize=30, rotation=45)
            plt.yticks(fontsize=30)
            plt.legend(fontsize=30, frameon=True, shadow=True)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            mosaic_plot_path = f"{report.temp_dir}/label_{label}_mosaic_evolution.png"
            plt.savefig(mosaic_plot_path)
            timepoint_graphs.append(mosaic_plot_path)
            plt.close()
    else:
        all_timepoint_df = current_df.copy()

    screenshot_path = screenshot_mosaic_blend(
        args.input_volume,
        args.input_labels,
        min_val=0,
        max_val=140,
        nb_rows=3,
        nb_columns=3,
        output_prefix="labels",
        directory=report.temp_dir,
    )

    report.render(current_df.to_dict("records"), screenshot_path, timepoint_graphs)
    report.to_pdf(args.output_report)
    if args.output_longitudinal:
        all_timepoint_df.drop(
            columns=[
                "label_name",
                "last_volume",
                "diff_volume",
                "diff_perc_volume",
                "last_volume_normalized",
                "diff_volume_normalized",
                "diff_perc_volume_normalized",
            ],
            errors="ignore",
        ).to_json(
            args.output_longitudinal, orient="records", indent=4, double_precision=15
        )
