import unittest
from unittest.mock import patch
from argparse import ArgumentParser, Namespace

import numpy as np

from avnirpy.io.utils import (
    add_verbose_arg,
    assert_inputs_exist,
    assert_outputs_exist,
    add_overwrite_arg,
)


class TestAddVerboseArg(unittest.TestCase):

    def setUp(self):
        self.parser = ArgumentParser()

    def test_default_verbose_level(self):
        add_verbose_arg(self.parser)
        args = self.parser.parse_args([])
        self.assertEqual(args.verbose, "WARNING")

    def test_verbose_level_info(self):
        add_verbose_arg(self.parser)
        args = self.parser.parse_args(["-v"])
        self.assertEqual(args.verbose, "INFO")

    def test_verbose_level_debug(self):
        add_verbose_arg(self.parser)
        args = self.parser.parse_args(["-v", "DEBUG"])
        self.assertEqual(args.verbose, "DEBUG")

    def test_verbose_level_warning(self):
        add_verbose_arg(self.parser)
        args = self.parser.parse_args(["-v", "WARNING"])
        self.assertEqual(args.verbose, "WARNING")

    def test_invalid_verbose_level(self):
        add_verbose_arg(self.parser)
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["-v", "INVALID"])


class TestAssertInputsExist(unittest.TestCase):

    def setUp(self):
        self.parser = ArgumentParser()

    @patch("os.path.isfile", return_value=True)
    def test_all_files_exist(self, mock_isfile):
        assert_inputs_exist(self.parser, ["file1.txt", "file2.txt"])
        # No exception should be raised

    @patch("os.path.isfile", return_value=False)
    def test_required_file_does_not_exist(self, mock_isfile):
        with self.assertRaises(SystemExit):
            assert_inputs_exist(self.parser, ["file1.txt"])

    @patch("os.path.isfile", side_effect=[True, False])
    def test_optional_file_does_not_exist(self, mock_isfile):
        with self.assertRaises(SystemExit):
            assert_inputs_exist(self.parser, ["file1.txt"], ["file2.txt"])

    @patch("os.path.isfile", side_effect=[True, True])
    def test_optional_file_exists(self, mock_isfile):
        assert_inputs_exist(self.parser, "file1.txt", "file2.txt")
        # No exception should be raised

    @patch("os.path.isfile", side_effect=[True, True, False])
    def test_mixed_files_exist(self, mock_isfile):
        with self.assertRaises(SystemExit):
            assert_inputs_exist(self.parser, ["file1.txt"], ["file2.txt", "file3.txt"])


class TestAddOverwriteArg(unittest.TestCase):

    def setUp(self):
        self.parser = ArgumentParser()

    def test_default_overwrite(self):
        add_overwrite_arg(self.parser)
        args = self.parser.parse_args([])
        self.assertFalse(args.overwrite)

    def test_overwrite_flag(self):
        add_overwrite_arg(self.parser)
        args = self.parser.parse_args(["-f"])
        self.assertTrue(args.overwrite)

    def test_help_message_without_delete_dirs(self):
        add_overwrite_arg(self.parser)
        help_message = self.parser.format_help()
        self.assertIn("Force overwriting of the output files.", help_message)
        self.assertNotIn(
            "CAREFUL.",
            help_message,
        )

    def test_help_message_with_delete_dirs(self):
        add_overwrite_arg(self.parser, will_delete_dirs=True)
        help_message = self.parser.format_help()
        self.assertIn("Force overwriting of the output files.", help_message)
        self.assertIn(
            "CAREFUL.",
            help_message,
        )


class TestAssertOutputsExist(unittest.TestCase):

    def setUp(self):
        self.parser = ArgumentParser()
        self.args = Namespace(overwrite=False)

    @patch("os.path.isfile", return_value=False)
    @patch("os.path.isdir", return_value=True)
    def test_all_files_do_not_exist(self, mock_isdir, mock_isfile):
        assert_outputs_exist(self.parser, self.args, ["dir1/", "file2.txt"])
        # No exception should be raised

    @patch("os.path.isfile", return_value=True)
    def test_required_file_exists_without_overwrite(self, mock_isfile):
        with self.assertRaises(SystemExit):
            assert_outputs_exist(self.parser, self.args, "file1.txt")

    @patch("os.path.isfile", side_effect=[False, True])
    def test_optional_file_exists_without_overwrite(self, mock_isfile):
        with self.assertRaises(SystemExit):
            assert_outputs_exist(self.parser, self.args, "file1.txt", "file2.txt")

    @patch("os.path.isfile", side_effect=[False, False])
    @patch("os.path.isdir", return_value=True)
    def test_optional_file_does_not_exist(self, mock_isdir, mock_isfile):
        assert_outputs_exist(self.parser, self.args, ["file1.txt"], ["file2.txt"])
        # No exception should be raised

    @patch("os.path.isfile", side_effect=[False, False])
    @patch("os.path.isdir", return_value=False)
    def test_directory_does_not_exist_with_check_dir_exists_false(
        self, mock_isdir, mock_isfile
    ):
        assert_outputs_exist(
            self.parser,
            self.args,
            ["file1.txt"],
            ["file2.txt"],
            check_dir_exists=False,
        )
        # No exception should be raised


if __name__ == "__main__":
    unittest.main()
