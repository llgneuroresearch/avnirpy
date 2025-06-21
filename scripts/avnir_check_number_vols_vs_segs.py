import os
import re
import pprint
import shutil
import argparse
import logging
import json
from glob import glob


def _get_arg_parser():
    """
    Build argparse.
    Returns: parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", default="config.yaml")
    parser.add_argument("-iv", "--input_directory_volumes",
                        help="Path to the directory containing the volumes")
    parser.add_argument("-is", "--input_directory_segmentations",
                        help="Path to the directory containing the segmentations")
    parser.add_argument("-o", "--output_directory", help="Path to the output directory",
                        default="same as input")
    parser.add_argument("-l", "--log_level", help="Level of the logging", default="debug")
    parser.add_argument("--regex_seg_id",
                        help="Regex to extract the id from the segmentation file name",
                        default='^([^_]+(?:_[^_]+)*)_[^_]+_')

    return parser


def _get_logger(args):
    """
    Get logger
    Args:
        args: argparse arguments
    Returns:
        logger: logger object
    """
    logger = logging.getLogger(__name__)
    # match the log level in argparse to the logging module
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
    fh = logging.FileHandler('avnir_check_number_of_files/checking_nb_files.log')
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
    """
    with open(output_file, "w") as f:
        json.dump(data, f)


def get_files(args):
    """
    Get the list of volume and mask files
    Args:
        args: argparser arguments
    Returns:
        vol_list: list of volume files
        mask_list: list of mask files
    """

    vol_list = sorted(glob(os.path.join(args.input_directory_volumes, '*')))
    vol_list = [vol for vol in vol_list if os.path.isfile(vol)] # remove directories

    mask_list = sorted(glob(os.path.join(args.input_directory_segmentations, '*')))
    # mask_list = [mask for mask in mask_list if os.path.isfile(mask)]

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
        vols = sorted(
            [vol for vol in vol_list if os.path.basename(vol).split('.')[1] in ["nrrd", "nii"]])
        print(f'len(vol_list): {len(vol_list)}')

    except IndexError as e:
        print(f'Error: {e}')


    try:
        masks = sorted(
            [mask for mask in mask_list if os.path.basename(mask).split('.')[1] in ["seg", "nii"]])
        print(f'len(mask_list): {len(mask_list)}')
    except IndexError as e:
        print(f'Error: {e}')
        print(f'len(mask_list): {len(mask_list)}')

    return vols, masks


def check_number_of_volumes_and_masks(vol_list, mask_list) :
    """
    Check if the number of volumes and masks match by number and ids
    Expects that the volume and mask file names are the same except for the extension.
    Args:
        vol_list (List[str]): List of volume filepaths.
        mask_list (List[str]): List of mask filepaths.


    Returns:
        bool : True if the number of volumes and masks match, False otherwise
        list : list of volume ids
        list : list of mask ids
    """

    try:
        assert len(vol_list) == len(mask_list)
        number_of_volumes_and_masks_match = True

    except AssertionError as e:
        number_of_volumes_and_masks_match = False
        print(e)

    return number_of_volumes_and_masks_match


def check_id_match(vol_list, mask_list, args):
    """
    Args:
        vol_list:
        mask_list:
        args:
    Returns:
        all_ids_match:
        unmatched_vol:
        unmatched_mask:
    """
    vol_ids = {os.path.basename(vol).split('.')[0] for vol in vol_list}
    print(f'len segm list: {len(mask_list)}')
    print(f'exemple segm list: {mask_list[0]}')
    mask_ids = {re.match(str(args.regex_seg_id), os.path.basename(mask)).group(1) for mask in mask_list}

    unmatched_vol = vol_ids - mask_ids
    unmatched_mask = mask_ids - vol_ids

    if not unmatched_vol and not unmatched_mask:
        all_ids_match = True
    else:
        all_ids_match = False

    return all_ids_match, unmatched_vol, unmatched_mask


def copy_matched_files_to_output(vol_list, mask_list, unmatched_vol, unmatched_mask, output_dir):
    """
    Copy the matched files to the output directory
    Args:
        vol_list (List[str]): List of volume filepaths.
        mask_list (List[str]): List of mask filepaths.
        unmatched_vol (set): set of unmatched volume ids
        unmatched_mask (set): set of unmatched mask ids
        output_dir (str): output directory
    """

    output_vol = os.path.join(output_dir, 'final_data', 'vols')
    if not os.path.exists(output_vol):
        os.makedirs(output_vol)

    output_unmatched_vol = os.path.join(output_dir, 'final_data', 'unmatched_vols')
    if not os.path.exists(output_unmatched_vol):
        os.makedirs(output_unmatched_vol)

    for vol in vol_list:
        id_case = os.path.basename(vol).split('.')[0]
        if id_case not in unmatched_vol:
            shutil.copy(vol, os.path.join(output_vol, os.path.basename(vol)))
        elif id_case in unmatched_vol:
            shutil.copy(vol, os.path.join(output_unmatched_vol, os.path.basename(vol)))


    output_mask = os.path.join(output_dir, 'final_data', 'masks')
    if not os.path.exists(output_mask):
        os.makedirs(output_mask)

    output_unmatched_mask = os.path.join(output_dir, 'final_data', 'unmatched_masks')
    if not os.path.exists(output_unmatched_mask):
        os.makedirs(output_unmatched_mask)

    for mask in mask_list:
        id_case = os.path.basename(mask).split('.')[0]
        if id_case not in unmatched_mask:
            shutil.copy(mask, os.path.join(output_mask, os.path.basename(mask)))
        elif id_case in unmatched_mask:
            shutil.copy(mask, os.path.join(output_unmatched_mask, os.path.basename(mask)))


def main():
    parser = _get_arg_parser()
    args = parser.parse_args()
    logger = _get_logger(args)
    logger.info('Starting script')
    logger.info('Reading config file')

    # Create output directory
    if args.output_directory == "same as input":
        args.output_directory = os.path.join(
            os.path.dirname(args.input_directory_volumes), 'output')
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
    logger.info(f'Output directory: {args.output_directory}')

    # gets all file in input directories
    vol_list, mask_list = get_files(args)
    logger.info(f'Found {len(vol_list)} volumes and {len(mask_list)} masks')

    # get only nifti or nrrd files from the list
    vols, masks = get_files_with_extension(vol_list, mask_list)
    logger.debug(f'Volume list exemple: {vols[0]}')
    logger.debug(f'Mask list exemple: {masks[0]}')

    # get boolean flags for number of volumes and masks match and all ids match
    number_of_volumes_and_masks_match = check_number_of_volumes_and_masks(vols, masks)
    if not number_of_volumes_and_masks_match:
        logger.warning(f"Number of volumes and masks do not match: "
                       f"{len(vols)} volumes vs {len(masks)} masks!")


    all_ids_match, unmatched_vol, unmatched_mask = check_id_match(vols, masks, args)
    if not all_ids_match:
        message = "Not all ids match between volumes and masks! \n"
        message += f"Unmatched volumes: {unmatched_vol} \n"
        message += f"Unmatched masks: {unmatched_mask} \n"
        message += "Writing unmatched files to json. \n"
        message += ("The script will ignore the unmatched files and "
                    "copy the matched files to the output directory")
        logger.warning(message)
        unmatched_vol = list(unmatched_vol)  # convert set to list to be able to dump to json
        unmatched_mask = list(unmatched_mask)
        _json_dump(unmatched_vol, os.path.join(args.output_directory, "unmatched_vol.json"))
        _json_dump(unmatched_mask, os.path.join(args.output_directory, "unmatched_mask.json"))

    # copy final dataset to output directory
    copy_matched_files_to_output(
        vols, masks, unmatched_vol, unmatched_mask, args.output_directory)
    logger.info(f'Moved final dataset to output directory {args.output_directory}')
    logger.info('You can now proceed with avnir_qc_labels '
                'to continue the segment level qc check.py')


if __name__ == "__main__":
    main()
