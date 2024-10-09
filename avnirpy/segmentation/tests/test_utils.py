import numpy as np
import pytest
from avnirpy.segmentation.utils import replace_labels_in_file


def test_replace_labels_in_file():
    label_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    label_nrrdhearder = {
        "Segment1_LabelValue": 1,
        "Segment2_LabelValue": 2,
        "Segment3_LabelValue": 3,
    }
    labels_in_file = {"Segment1": 1, "Segment2": 2, "Segment3": 3}
    labels_in_config = {"Segment1": 10, "Segment2": 20, "Segment3": 3}
    segment_match = {
        "Segment1": "Segment1",
        "Segment2": "Segment2",
        "Segment3": "Segment3",
    }

    expected_label_data = np.array([[10, 20, 3], [4, 5, 6], [7, 8, 9]])
    expected_label_nrrdhearder = {
        "Segment1_LabelValue": 10,
        "Segment2_LabelValue": 20,
        "Segment3_LabelValue": 3,
    }

    new_label_data, new_label_nrrdhearder = replace_labels_in_file(
        label_data, label_nrrdhearder, labels_in_file, labels_in_config, segment_match
    )

    assert np.array_equal(new_label_data, expected_label_data)
    assert new_label_nrrdhearder == expected_label_nrrdhearder


def test_replace_labels_in_file_no_change():
    label_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    label_nrrdhearder = {
        "Segment1_LabelValue": 1,
        "Segment2_LabelValue": 2,
        "Segment3_LabelValue": 3,
    }
    labels_in_file = {"Segment1": 1, "Segment2": 2, "Segment3": 3}
    labels_in_config = {"Segment1": 1, "Segment2": 2, "Segment3": 3}
    segment_match = {
        "Segment1": "Segment1",
        "Segment2": "Segment2",
        "Segment3": "Segment3",
    }

    expected_label_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_label_nrrdhearder = {
        "Segment1_LabelValue": 1,
        "Segment2_LabelValue": 2,
        "Segment3_LabelValue": 3,
    }

    new_label_data, new_label_nrrdhearder = replace_labels_in_file(
        label_data, label_nrrdhearder, labels_in_file, labels_in_config, segment_match
    )

    assert np.array_equal(new_label_data, expected_label_data)
    assert new_label_nrrdhearder == expected_label_nrrdhearder


def test_replace_labels_in_file_partial_change():
    label_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    label_nrrdhearder = {
        "Segment1_LabelValue": 1,
        "Segment2_LabelValue": 2,
        "Segment3_LabelValue": 3,
    }
    labels_in_file = {"Segment1": 1, "Segment2": 2, "Segment3": 3}
    labels_in_config = {"Segment1": 10, "Segment2": 2, "Segment3": 30}
    segment_match = {
        "Segment1": "Segment1",
        "Segment2": "Segment2",
        "Segment3": "Segment3",
    }

    expected_label_data = np.array([[10, 2, 30], [4, 5, 6], [7, 8, 9]])
    expected_label_nrrdhearder = {
        "Segment1_LabelValue": 10,
        "Segment2_LabelValue": 2,
        "Segment3_LabelValue": 30,
    }

    new_label_data, new_label_nrrdhearder = replace_labels_in_file(
        label_data, label_nrrdhearder, labels_in_file, labels_in_config, segment_match
    )

    assert np.array_equal(new_label_data, expected_label_data)
    assert new_label_nrrdhearder == expected_label_nrrdhearder
