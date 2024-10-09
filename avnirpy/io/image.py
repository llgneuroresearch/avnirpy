import nibabel as nib
from nibabel.nifti1 import Nifti1Header
import nrrd
from nrrd.types import NRRDHeader
import numpy as np
from typing import List, Tuple

SPACE_CONVERTER = {
    "L": "left",
    "R": "right",
    "P": "posterior",
    "A": "anterior",
    "I": "inferior",
    "S": "superior",
}


def axcode_vector(axcode: List[str]) -> np.ndarray:
    """
    Generate a diagonal matrix representing the transformation for a given axis direction code.
    For LPI, the transformation matrix is a diagonal of [-1, -1, -1, 1].

    Parameters:
        axcode (List[str]): A list of three characters representing the axis direction code.

    Returns:
        numpy.ndarray: A diagonal matrix representing the transformation for the given axis
        direction code.
    """
    xfrm = [-1, -1, -1, 1]
    for i, (code, to_ax) in enumerate(zip(axcode, ["L", "P", "I"])):
        if code != to_ax:
            xfrm[i] = 1

    return np.diag(xfrm)


def load_nrrd(
    nrrd_image: str,
) -> Tuple[np.ndarray, Nifti1Header, NRRDHeader, np.ndarray]:
    """
    Load a NRRD image file.

    Parameters:
        nrrd_image (str): The path to the NRRD image file.

    Returns:
        numpy.ndarray: The image data.
        Nifti1Header: The NIfTI header.
        NRRDHeader: The NRRD header.
        numpy.ndarray: The affine transformation matrix.
    """
    img = nrrd.read(nrrd_image)
    nrrd_header = img[1]

    translation = nrrd_header["space origin"]
    rotation = nrrd_header["space directions"]
    affine_nhdr = np.vstack(
        (np.hstack((rotation.T, np.reshape(translation, (3, 1)))), [0, 0, 0, 1])
    )
    axcode = nib.orientations.aff2axcodes(affine_nhdr)
    axcode = nrrd_header["space"].split("-")
    axcode = [code[0].capitalize() for code in axcode]
    transform = axcode_vector(axcode)
    affine = np.dot(transform, affine_nhdr)

    nii_header = Nifti1Header()
    nii_header.set_xyzt_units(xyz=2, t=0)
    nii_header["qform_code"] = 1
    nii_header["sform_code"] = 1

    return img[0], nii_header, nrrd_header, affine


def write_nrrd(
    nrrd_image: str, data: np.ndarray, affine: np.ndarray, header: dict = {}
) -> None:
    """
    Write a NRRD image file.

    Parameters:
        nrrd_image (str): The path to save the NRRD image file.
        data (numpy.ndarray): The image data.
        affine (numpy.ndarray): The affine transformation matrix.
        header (dict): The NRRD header.
    """
    axcode = nib.orientations.aff2axcodes(affine)
    transform = axcode_vector(axcode)
    affine = np.dot(transform, affine)

    header["space origin"] = affine[:3, 3]
    header["space directions"] = affine[:3, :3].T
    header["space"] = (
        f"{SPACE_CONVERTER[axcode[0]]}-{SPACE_CONVERTER[axcode[1]]}-{SPACE_CONVERTER[axcode[2]]}"
    )

    nrrd.write(nrrd_image, data, header)


def load_nifti(nifti_image: str) -> Tuple[np.ndarray, Nifti1Header, np.ndarray]:
    """
    Load a NIfTI image file.

    Parameters:
        nifti_image (str): The path to the NIfTI image file.

    Returns:
        numpy.ndarray: The image data.
        Nifti1Header: The NIfTI header.
        numpy.ndarray: The affine transformation matrix.
    """
    img = nib.load(nifti_image)
    affine = img.affine
    nii_header = img.header

    return img.get_fdata(), nii_header, affine


def get_labels_from_nrrd_header(nrrd_header: NRRDHeader) -> Tuple[dict, dict]:
    """
    Extract the labels from the NRRD header.

    Parameters:
        nrrd_header (NRRDHeader): The NRRD header.

    Returns:
        dict: A dictionary mapping the label ID to the label name.
        dict: A dictionary mapping the label name to the segment ID.
    """
    label_in_file = {}
    segment_match = {}
    segments = []
    for key in nrrd_header:
        if "Segment" in key and "_ID" in key:
            segments.append(key.split("_")[0])

    for segment in segments:
        label_in_file[nrrd_header[f"{segment}_Name"]] = int(
            nrrd_header[f"{segment}_LabelValue"]
        )
        segment_match[nrrd_header[key].split("_")[-1]] = segment

    return label_in_file, segment_match
