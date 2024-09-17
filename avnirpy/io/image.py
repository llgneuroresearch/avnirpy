import nibabel as nib
from nibabel.nifti1 import Nifti1Header
import nrrd
import numpy as np
from typing import List, Tuple


def axcode_transform(axcode: List[str], to_axcode: List[str]) -> np.ndarray:
    """
    Converts the axis direction code to another coordinate system.

    Parameters:
        axcode (List[str]): A list of three characters representing the axis direction code.
        to_axcode (List[str]): A list of three characters representing the target axis direction code.

    Returns:
        numpy.ndarray: A diagonal matrix representing the transformation from the axis direction
                       code to the target coordinate system.
    """
    xfrm = [1, 1, 1, 1]
    for i, (code, to_ax) in enumerate(zip(axcode, to_axcode)):
        if code != to_ax:
            xfrm[i] = -1

    return np.diag(xfrm)


def load_nrrd(nrrd_image: str) -> Tuple[np.ndarray, Nifti1Header, np.ndarray]:
    """
    Load a NRRD image file.

    Parameters:
        nrrd_image (str): The path to the NRRD image file.

    Returns:
        numpy.ndarray: The image data.
        Nifti1Header: The NIfTI header.
        numpy.ndarray: The affine transformation matrix.
    """
    img = nrrd.read(nrrd_image)
    hdr = img[1]

    translation = hdr["space origin"]
    rotation = hdr["space directions"]
    affine_nhdr = np.vstack(
        (np.hstack((rotation.T, np.reshape(translation, (3, 1)))), [0, 0, 0, 1])
    )
    axcode = nib.orientations.aff2axcodes(affine_nhdr)
    to_ras = axcode_transform(axcode, ["R", "A", "S"])
    affine = np.dot(to_ras, affine_nhdr)

    nii_header = Nifti1Header()
    nii_header.set_xyzt_units(xyz=2, t=0)
    nii_header["qform_code"] = 1
    nii_header["sform_code"] = 1

    return img[0], nii_header, affine


def write_nrrd(nrrd_image: str, data: np.ndarray, affine: np.ndarray) -> None:
    """
    Write a NRRD image file.

    Parameters:
        nrrd_image (str): The path to save the NRRD image file.
        data (numpy.ndarray): The image data.
        header (dict): The NRRD header.
    """
    axcode = nib.orientations.aff2axcodes(affine)
    to_ras = axcode_transform(axcode, ["R", "A", "S"])
    affine = np.dot(to_ras, affine)

    header = {}
    header["space origin"] = affine[:3, 3]
    header["space directions"] = affine[:3, :3].T
    header["space"] = "right-anterior-superior"

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
