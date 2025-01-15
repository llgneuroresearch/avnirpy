import os
import numpy as np
import nibabel as nib
from PIL import Image
from unittest import mock
import pytest

from avnirpy.reporting.screenshot import (
    screenshot_mosaic_wrapper,
    screenshot_mosaic_blend,
    screenshot_mosaic,
)

# Mock data for testing
mock_data = np.random.randint(5, size=(100, 100, 100))


@pytest.fixture
def mock_nib_load():
    with mock.patch("nibabel.load") as mock_load:
        mock_img = mock.Mock()
        mock_img.get_fdata.return_value = mock_data
        mock_load.return_value = mock_img
        yield mock_load


@pytest.fixture
def mock_image_save():
    with mock.patch.object(Image.Image, "save") as mock_save:
        yield mock_save


def test_screenshot_mosaic_wrapper(mock_nib_load, mock_image_save):
    filename = "test_image.nii"
    output_prefix = "test_prefix"
    directory = "."
    result = screenshot_mosaic_wrapper(filename, output_prefix, directory)
    assert result == os.path.join(directory, output_prefix + "_test_image.png")
    mock_image_save.assert_called_once()


def test_screenshot_mosaic_wrapper_no_return_path(mock_nib_load):
    filename = "test_image.nii"
    result = screenshot_mosaic_wrapper(filename, return_path=False)
    assert isinstance(result, Image.Image)


def test_screenshot_mosaic_blend(mock_nib_load, mock_image_save):
    image = "test_image.nii"
    image_blend = "test_image_blend.nii"
    output_prefix = "test_prefix"
    directory = "."
    result = screenshot_mosaic_blend(image, image_blend, output_prefix, directory)
    assert result == os.path.join(directory, output_prefix + "_test_image.png")
    mock_image_save.assert_called_once()


def test_screenshot_mosaic():
    data = mock_data
    skip = 1
    pad = 20
    nb_columns = 15
    result = screenshot_mosaic(data, skip, pad, nb_columns)
    assert isinstance(result, Image.Image)


def test_screenshot_anatomy_mosaic():
    data = np.random.randint(1000, size=(100, 100, 100))
    skip = 1
    pad = 20
    nb_columns = 2
    min_val = 60
    max_val = 200
    result = screenshot_mosaic(data, skip, pad, nb_columns, min_val, max_val)
    assert isinstance(result, Image.Image)
