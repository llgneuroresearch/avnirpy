import numpy as np
from unittest import mock
import nibabel as nib
from avnirpy.io.image import axcode_transform, load_nrrd, write_nrrd, load_nifti
from avnirpy.io.image import get_labels_from_nrrd_header


def test_axcode_transform():
    axcode = ["L", "P", "I"]
    to_axcode = ["R", "A", "S"]
    expected = np.diag([-1, -1, -1, 1])
    result = axcode_transform(axcode, to_axcode)
    np.testing.assert_array_equal(result, expected)


@mock.patch("nrrd.read")
@mock.patch("nibabel.orientations.aff2axcodes")
def test_load_nrrd(mock_aff2axcodes, mock_nrrd_read):
    mock_nrrd_read.return_value = (
        np.zeros((10, 10, 10)),
        {"space origin": [0, 0, 0], "space directions": np.eye(3)},
    )
    mock_aff2axcodes.return_value = ["R", "A", "S"]

    data, nii_header, nrdd_header, affine = load_nrrd("dummy_path")

    assert data.shape == (10, 10, 10)
    assert isinstance(nii_header, nib.Nifti1Header)
    assert isinstance(nrdd_header, dict)
    assert affine.shape == (4, 4)


@mock.patch("nrrd.write")
@mock.patch("nibabel.orientations.aff2axcodes")
def test_write_nrrd(mock_aff2axcodes, mock_nrrd_write):
    mock_aff2axcodes.return_value = ["R", "A", "S"]
    data = np.zeros((10, 10, 10))
    affine = np.eye(4)

    write_nrrd("dummy_path", data, affine)

    mock_nrrd_write.assert_called_once()
    args, kwargs = mock_nrrd_write.call_args
    assert args[0] == "dummy_path"
    np.testing.assert_array_equal(args[1], data)
    assert "space origin" in args[2]
    assert "space directions" in args[2]
    assert args[2]["space"] == "right-anterior-superior"


@mock.patch("nibabel.load")
def test_load_nifti(mock_nib_load):
    mock_img = mock.Mock()
    mock_img.get_fdata.return_value = np.zeros((10, 10, 10))
    mock_img.affine = np.eye(4)
    mock_img.header = nib.Nifti1Header()
    mock_nib_load.return_value = mock_img

    data, header, affine = load_nifti("dummy_path")

    assert data.shape == (10, 10, 10)
    assert isinstance(header, nib.Nifti1Header)
    assert affine.shape == (4, 4)


def test_get_labels_from_nrrd_header():
    nrrd_header = {
        "Segment1_ID": "Segment_1",
        "Segment1_LabelValue": "1",
        "Segment2_ID": "Segment_2",
        "Segment2_LabelValue": "2",
    }
    expected_labels = {
        1: "1",
        2: "2",
    }
    result = get_labels_from_nrrd_header(nrrd_header)
    assert result == expected_labels


def test_get_labels_from_nrrd_header_empty():
    nrrd_header = {}
    expected_labels = {}
    result = get_labels_from_nrrd_header(nrrd_header)
    assert result == expected_labels


def test_get_labels_from_nrrd_header_partial():
    nrrd_header = {
        "Segment1_ID": "Segment_1",
        "Segment1_LabelValue": "1",
    }
    expected_labels = {
        1: "1",
    }
    result = get_labels_from_nrrd_header(nrrd_header)
    assert result == expected_labels
