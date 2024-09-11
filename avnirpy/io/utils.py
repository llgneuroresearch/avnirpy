import os
from typing import Union, List

from argparse import ArgumentParser, Namespace
import numpy as np


def add_verbose_arg(parser: ArgumentParser) -> None:
    """**Imported from Scilpy**
    Add verbose option to the parser

    Args:
        parser (ArgumentParser): Argument Parser
    """
    parser.add_argument(
        "-v",
        default="WARNING",
        const="INFO",
        nargs="?",
        choices=["DEBUG", "INFO", "WARNING"],
        dest="verbose",
        help="Produces verbose output depending on "
        "the provided level. \nDefault level is warning, "
        "default when using -v is info.",
    )


def assert_inputs_exist(
    parser: ArgumentParser,
    required: Union[str, List[str]],
    optional: Union[str, List[str]] = None,
) -> None:
    """**Imported from Scilpy**
    Assert that all inputs exist. If not, print parser's usage and exit.

    Args:
        parser (ArgumentParser): Parser.
        required (Union[str, List[str]]): Required paths to be checked.
        optional (Union[str, List[str]], optional): Optional paths to be checked. Defaults to None.
    """

    def _check(path: str):
        """Check if file exists.

        Args:
            path (str): filename
        """
        if not os.path.isfile(path):
            parser.error("Input file {} does not exist".format(path))

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        _check(required_file)
    if optional is not None:
        for optional_file in optional:
            _check(optional_file)


def add_overwrite_arg(parser: ArgumentParser, will_delete_dirs: bool = False) -> None:
    """**Imported from Scilpy**
    Add overwrite option to the parser

    Args:
        parser (ArgumentParser): Parser.
        will_delete_dirs (bool, optional): Delete the directory to overwrite. Defaults to False.
    """
    if will_delete_dirs:
        _help = (
            "Force overwriting of the output files.\n"
            "CAREFUL. The whole output directory will be deleted if it "
            "exists."
        )
    else:
        _help = "Force overwriting of the output files."
    parser.add_argument("-f", dest="overwrite", action="store_true", help=_help)


def assert_outputs_exist(
    parser: ArgumentParser,
    args: Namespace,
    required: Union[str, List[str]],
    optional: Union[str, List[str]] = None,
    check_dir_exists: bool = True,
) -> None:
    """**Imported from Scilpy**
    Assert that all outputs don't exist or that if they exist, -f was used.
    If not, print parser's usage and exit.

    Args:
        parser (ArgumentParser): Parser.
        args (Namespace): Argument list.
        required (Union[str, List[str]]): Required paths to be checked.
        optional (Union[str, List[str]], optional): Optional paths to be checked. Defaults to None.
        check_dir_exists (bool, optional): Test if output directory exists. Defaults to True.
    """

    def check(path: str):
        """Check if file or diectory exists.

        Args:
            path (str): file or directory name
        """
        if os.path.isfile(path) and not args.overwrite:
            parser.error(
                "Output file {} exists. Use -f to force " "overwriting".format(path)
            )

        if check_dir_exists:
            path_dir = os.path.dirname(path)
            if path_dir and not os.path.isdir(path_dir):
                parser.error(
                    "Directory {}/ \n for a given output file "
                    "does not exists.".format(path_dir)
                )

    if isinstance(required, str):
        required = [required]

    if isinstance(optional, str):
        optional = [optional]

    for required_file in required:
        check(required_file)
    for optional_file in optional or []:
        check(optional_file)
