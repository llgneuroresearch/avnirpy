import os, re, shutil
import argparse
import logging
import yaml
import json
from glob import glob
from time import time



def _get_arg_parser():
    """
    Build argparser.
    Returns: parser
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", default="config.yaml")
    parser.add_argument("-iv", "--input_directory_volumes", help="Path to the directory containing the volumes")
    parser.add_argument("-is", "--input_directory_segmentations", help="Path to the directory containing the segmentations")
    parser.add_argument("-o", "--output_directory", help="Path to the output directory", default="same as input")
    parser.add_argument("-l", "--log_level", help="Level of the logging", default="debug")

    return parser

def _get_config(args):
    """
    Read the yaml file and return the config dictionary
    Args:
        args: argparser arguments
    Returns:
        config: dictionary
    """
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return config

def _get_logger(args):
    """
    Get logger
    Args:
        args: argparser arguments
    Returns:
        logger: logger object
    """
    logger = logging.getLogger(__name__)
    # match the log level in argparser to the logging module
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    level = levels.get(args.log_level.lower())
    logger.setLevel(level)

    # create file handler which logs even debug messages
    fh = logging.FileHandler('checking_nb_files.log')
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def _json_dump(data, output_file):
    """
    Dump data to a json file
    Args:
        data: data to dump
        output_file: path to the output file
    """
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)


def get_files(args):
    """
    Get the list of volume and mask files
    Args:
        args: argparser arguments
    Returns:
        vol_list: list of volume files
        mask_list: list of mask files
    """

    vol_list = sorted(glob(os.path.join(args.input_directory_volumes, "*")))
    mask_list = sorted(glob(os.path.join(args.input_directory_segmentations, "*")))

    return vol_list, mask_list


def get_files_with_extension(vol_list, mask_list):
    """
    Check if the extension of the files is nrrd or nifti
    Args:
        vol_list: list of volume files
        mask_list: list of mask files
    Returns:
        list : list containig only the volume files
        list : list containig only the mask files
    """

    try:
        vol_list = sorted([vol for vol in vol_list if os.path.basename(vol).split('.')[1] in ["nrrd", "nii"]])
        print(f'len(vol_list): {len(vol_list)}')
    except IndexError as e:
        print(f'Error: {e}')
        print(f'len(vol_list): {len(vol_list)}')

    try:
        mask_list = sorted([mask for mask in mask_list if os.path.basename(mask).split('.')[1] in ["seg", "nii"]])
        print(f'len(mask_list): {len(mask_list)}')
    except IndexError as e:
        print(f'Error: {e}')
        print(f'len(mask_list): {len(mask_list)}')

    return vol_list, mask_list


def check_number_of_volumes_and_masks(vol_list, mask_list) -> bool:
    """
    Check if the number of volumes and masks match by number and ids

    Args:
        vol_list (List[str]): List of volume filepaths.
        mask_list (List[str]): List of mask filepaths.
        note: expects that the volume and mask file names are the same except for the extension.

    Returns:
        bool : True if the number of volumes and masks match, False otherwise.
        list : list of volume ids
        list : list of mask ids
    """
    vol_filenames = [os.path.basename(vol) for vol in vol_list]
    mask_filenames = [os.path.basename(mask) for mask in mask_list]
    try:
        vol_ids = [os.path.basename(vol).split('.')[0] for vol in vol_filenames]
    except IndexError as e:
        print(f'Error: {e}')

    try:
        mask_ids = [os.path.basename(mask).split('.')[0] for mask in mask_filenames]
    except IndexError as e:
        print(f'Error: {e}')


    print(f"Number of volumes: {len(vol_list)}")
    print(f"Number of masks: {len(mask_list)}")
    try:
        assert len(vol_list) == len(mask_list)
        number_of_volumes_and_masks_match = True

    except AssertionError as e:
        number_of_volumes_and_masks_match = False

    print(f"Number of volumes and masks match: {number_of_volumes_and_masks_match}")

    return number_of_volumes_and_masks_match, vol_ids, mask_ids

def check_id_match(vol_list, mask_list):
    vol_ids = {os.path.basename(vol).split('.')[0] for vol in vol_list}
    mask_ids = {os.path.basename(mask).split('.')[0] for mask in mask_list}

    unmatched_vol = vol_ids - mask_ids
    unmatched_mask = mask_ids - vol_ids

    if not unmatched_vol and not unmatched_mask:
        all_ids_match = True
    else:
        all_ids_match = False

    return  all_ids_match, unmatched_vol, unmatched_mask


def copy_matched_files_to_output(vol_list, mask_list, unmatched_vol, unmatched_mask, output_dir):
    """
    Copy the matched files to the output directory
    Args:
        vol_list (List[str]): List of volume filepaths.
        mask_list (List[str]): List of mask filepaths.
        unmatched_files (List[str]): List of unmatched file ids.
        output_dir (str): Path to the output directory.
    """

    output_vol = os.path.join(output_dir, 'final_data', 'vols')
    if not os.path.exists(output_vol):
        os.makedirs(output_vol)
    for vol in vol_list:
        id = os.path.basename(vol).split('.')[0]
        if id not in unmatched_vol:
            shutil.copy(vol, os.path.join(output_vol, os.path.basename(vol)))

    output_mask = os.path.join(output_dir, 'final_data', 'masks')
    if not os.path.exists(output_mask):
        os.makedirs(output_mask)
    for mask in mask_list:
        id = os.path.basename(mask).split('.')[0]
        if id not in unmatched_mask:
            shutil.copy(mask, os.path.join(output_mask, os.path.basename(mask)))


def main():
    parser = _get_arg_parser()
    args = parser.parse_args()
    logger = _get_logger(args)
    logger.info('Starting script')
    logger.info('Reading config file')
    config = _get_config(args)
    logger.debug(f'Config: {config}')

    #Create output directory
    if args.output_directory == "same as input":
        args.output_directory= os.path.join(os.path.dirname(args.input_directory_volumes), 'output')
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
    logger.info(f'Output directory: {args.output_directory}')

    # gets all file in input directories
    vol_list, mask_list = get_files(args)
    logger.info(f'Found {len(vol_list)} volumes and {len(mask_list)} masks')
    # get only nifti or nrrd files from the list
    vol_list, mask_list = get_files_with_extension(vol_list, mask_list)
    logger.debug(f'Volume list exemple: {vol_list[0]}')
    logger.debug(f'Mask list exemple: {mask_list[0]}')

    # get boolean flags for number of volumes and masks match and all ids match
    number_of_volumes_and_masks_match, vol_ids, mask_ids = check_number_of_volumes_and_masks(vol_list, mask_list)
    if not number_of_volumes_and_masks_match:
        logger.warning(f"Number of volumes and masks do not match: {len(vol_list)} volumes vs {len(mask_list)} masks!")

    all_ids_match, unmatched_vol, unmatched_mask = check_id_match(vol_ids, mask_ids)
    if not all_ids_match:
        message = "Not all ids match between volumes and masks! \n"
        message += f"Unmatched volumes: {unmatched_vol} \n"
        message += f"Unmatched masks: {unmatched_mask} \n"
        message += "Writing unmatched files to json. \n"
        message += "The script will ignore the unmatched files and copy the matched files to the output directory"
        logger.warning(message)
        unmatched_vol = list(unmatched_vol) # convert set to list to be able to dump to json
        unmatched_mask = list(unmatched_mask)
        _json_dump(unmatched_vol, os.path.join(args.output_directory, "unmatched_vol.json"))
        _json_dump(unmatched_mask, os.path.join(args.output_directory, "unmatched_mask.json"))

    # copy final dataset to output directory
    copy_matched_files_to_output(vol_list, mask_list, unmatched_vol, unmatched_mask, args.output_directory)
    logger.info(f'Moved final dataset to output directory {args.output_directory}')
    logger.info('You can now proceed with avnir_qc_labels.py')

if __name__ == "__main__":
    main()