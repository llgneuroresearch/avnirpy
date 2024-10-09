import logging
from typing import Tuple

import numpy as np


def replace_labels_in_file(
    label_data: np.ndarray,
    label_nrrdhearder: dict,
    labels_in_file: dict,
    labels_in_config: dict,
    segment_match: dict,
) -> Tuple[np.ndarray, dict]:
    """
    Replace the labels in the file by the labels in the config file.

    Args:
        label_data (np.ndarray): The label data array.
        label_nrrdhearder (dict): The NRRD header of the label data.
        labels_in_file (dict): Dictionary of labels in the file.
        labels_in_config (dict): Dictionary of labels in the config file.
        segment_match (dict): Dictionary mapping segment names to their match in the file.
    """
    for name, label in labels_in_config.items():
        if name in labels_in_file and label != labels_in_file[name]:
            label_data[label_data == labels_in_file[name]] = label
            label_nrrdhearder[f"{segment_match[name]}_LabelValue"] = label
            logging.warning(
                f"Label {name} ({labels_in_file[name]}) has a different value in the config "
                f"file ({label}). The NRRD file is modified."
            )
    return label_data, label_nrrdhearder
