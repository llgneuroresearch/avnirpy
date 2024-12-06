import pytest
from unittest.mock import patch
from argparse import ArgumentParser, Namespace

import numpy as np
from avnirpy.version import __version__

from avnirpy.io.utils import (
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    add_overwrite_arg,
    check_images_space,
    check_segment_extent,
    add_version_arg,
)


@pytest.fixture
def parser():
    return ArgumentParser()


def test_default_verbose_level(parser):
    add_verbose_arg(parser)
    args = parser.parse_args([])
    assert args.verbose == "WARNING"


def test_verbose_level_info(parser):
    add_verbose_arg(parser)
    args = parser.parse_args(["-v"])
    assert args.verbose == "INFO"


def test_verbose_level_debug(parser):
    add_verbose_arg(parser)
    args = parser.parse_args(["-v", "DEBUG"])
    assert args.verbose == "DEBUG"


def test_verbose_level_warning(parser):
    add_verbose_arg(parser)
    args = parser.parse_args(["-v", "WARNING"])
    assert args.verbose == "WARNING"


def test_invalid_verbose_level(parser):
    add_verbose_arg(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["-v", "INVALID"])


@patch("os.path.isfile", return_value=True)
def test_all_files_exist(mock_isfile, parser):
    assert_inputs_exist(parser, ["file1.txt", "file2.txt"])
    # No exception should be raised


@patch("os.path.isfile", return_value=False)
def test_required_file_does_not_exist(mock_isfile, parser):
    with pytest.raises(SystemExit):
        assert_inputs_exist(parser, ["file1.txt"])


@patch("os.path.isfile", side_effect=[True, False])
def test_optional_file_does_not_exist(mock_isfile, parser):
    with pytest.raises(SystemExit):
        assert_inputs_exist(parser, ["file1.txt"], ["file2.txt"])


@patch("os.path.isfile", side_effect=[True, True])
def test_optional_file_exists(mock_isfile, parser):
    assert_inputs_exist(parser, "file1.txt", "file2.txt")
    # No exception should be raised


@patch("os.path.isfile", side_effect=[True, True, False])
def test_mixed_files_exist(mock_isfile, parser):
    with pytest.raises(SystemExit):
        assert_inputs_exist(parser, ["file1.txt"], ["file2.txt", "file3.txt"])


def test_default_overwrite(parser):
    add_overwrite_arg(parser)
    args = parser.parse_args([])
    assert not args.overwrite


def test_overwrite_flag(parser):
    add_overwrite_arg(parser)
    args = parser.parse_args(["-f"])
    assert args.overwrite


def test_help_message_without_delete_dirs(parser):
    add_overwrite_arg(parser)
    help_message = parser.format_help()
    assert "Force overwriting of the output files." in help_message
    assert "CAREFUL." not in help_message


def test_help_message_with_delete_dirs(parser):
    add_overwrite_arg(parser, will_delete_dirs=True)
    help_message = parser.format_help()
    assert "Force overwriting of the output files." in help_message
    assert "CAREFUL." in help_message


@pytest.fixture
def args():
    return Namespace(overwrite=False)


@patch("os.path.isfile", return_value=False)
@patch("os.path.isdir", return_value=True)
def test_all_files_do_not_exist(mock_isdir, mock_isfile, parser, args):
    assert_outputs_exist(parser, args, ["dir1/", "file2.txt"])
    # No exception should be raised


@patch("os.path.isfile", return_value=True)
def test_required_file_exists_without_overwrite(mock_isfile, parser, args):
    with pytest.raises(SystemExit):
        assert_outputs_exist(parser, args, "file1.txt")


@patch("os.path.isfile", side_effect=[False, True])
def test_optional_file_exists_without_overwrite(mock_isfile, parser, args):
    with pytest.raises(SystemExit):
        assert_outputs_exist(parser, args, "file1.txt", "file2.txt")


@patch("os.path.isfile", side_effect=[False, False])
@patch("os.path.isdir", return_value=True)
def test_output_optional_file_does_not_exist(mock_isdir, mock_isfile, parser, args):
    assert_outputs_exist(parser, args, ["file1.txt"], ["file2.txt"])
    # No exception should be raised


@patch("os.path.isfile", side_effect=[False, False])
@patch("os.path.isdir", return_value=False)
def test_directory_does_not_exist_with_check_dir_exists_false(
    mock_isdir, mock_isfile, parser, args
):
    assert_outputs_exist(
        parser,
        args,
        ["file1.txt"],
        ["file2.txt"],
        check_dir_exists=False,
    )
    # No exception should be raised


@pytest.fixture
def vol_header():
    return {
        "space origin": np.array([0.0, 0.0, 0.0]),
        "space directions": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "space": "left-posterior-superior",
    }


@pytest.fixture
def labels_header():
    return {
        "space origin": np.array([0.0, 0.0, 0.0]),
        "space directions": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "space": "left-posterior-superior",
    }


def test_same_space(vol_header, labels_header):
    assert check_images_space(vol_header, labels_header)


def test_different_space_origin(vol_header, labels_header):
    labels_header["space origin"] = np.array([1.0, 0.0, 0.0])
    assert not check_images_space(vol_header, labels_header)


def test_different_space_directions(vol_header, labels_header):
    labels_header["space directions"] = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    assert not check_images_space(vol_header, labels_header)


def test_different_space(vol_header, labels_header):
    labels_header["space"] = "right-anterior-superior"
    assert not check_images_space(vol_header, labels_header)


@pytest.fixture
def nrrd_header_consistent():
    return {
        "Segment1_Extend": "value1",
        "Segment2_Extend": "value1",
        "Segment3_Extend": "value1",
    }


@pytest.fixture
def nrrd_header_inconsistent():
    return {
        "Segment1_Extend": "value1",
        "Segment2_Extend": "value2",
        "Segment3_Extend": "value1",
    }


@pytest.fixture
def nrrd_header_no_segments():
    return {
        "SomeOtherKey": "value1",
        "AnotherKey": "value2",
    }


def test_check_segment_extent_consistent(nrrd_header_consistent):
    assert check_segment_extent(nrrd_header_consistent)


def test_check_segment_extent_inconsistent(nrrd_header_inconsistent):
    assert not check_segment_extent(nrrd_header_inconsistent)


def test_check_segment_extent_no_segments(nrrd_header_no_segments):
    assert check_segment_extent(nrrd_header_no_segments)


def test_version_arg(parser, capsys):
    add_version_arg(parser)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    assert __version__ == captured.out.strip()
