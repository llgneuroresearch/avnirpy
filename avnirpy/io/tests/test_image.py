import pytest
import numpy as np
from unittest import mock
from avnirpy.io.image import axcode_to_ras, nifti_write
import nibabel as nib
import nrrd
import os


def test_axcode_to_ras():
    # Test case where axcode is already in RAS
    axcode = ["R", "A", "S"]
    expected_output = np.diag([1, 1, 1, 1])
    assert np.array_equal(axcode_to_ras(axcode), expected_output)

    # Test case where axcode is in LPI (Left, Posterior, Inferior)
    axcode = ["L", "P", "I"]
    expected_output = np.diag([-1, -1, -1, 1])
    assert np.array_equal(axcode_to_ras(axcode), expected_output)

    # Test case where axcode is in RPI (Right, Posterior, Inferior)
    axcode = ["R", "P", "I"]
    expected_output = np.diag([1, -1, -1, 1])
    assert np.array_equal(axcode_to_ras(axcode), expected_output)


@mock.patch("nrrd.read")
@mock.patch("nibabel.save")
def test_nifti_write(mock_nib_save, mock_nrrd_read):
    # Mock the nrrd.read function
    mock_nrrd_read.return_value = (
        np.random.rand(5, 5, 5),  # Mock image data
        {"space origin": [0, 0, 0], "space directions": np.eye(3)},
    )

    # Define the input and output file paths
    nrrd_image = "input.nrrd"
    output = "output.nii"

    # Call the nifti_write function
    nifti_write(nrrd_image, output)

    # Check that nrrd.read was called with the correct arguments
    mock_nrrd_read.assert_called_once_with(nrrd_image)

    # Check that nib.save was called with the correct arguments
    assert mock_nib_save.call_count == 1
    saved_img = mock_nib_save.call_args[0][0]
    saved_output = mock_nib_save.call_args[0][1]

    assert saved_output == output
    assert isinstance(saved_img, nib.nifti1.Nifti1Image)
    assert saved_img.header["qform_code"] == 1
    assert saved_img.header["sform_code"] == 1
    assert saved_img.header.get_xyzt_units() == ("mm", "unknown")
