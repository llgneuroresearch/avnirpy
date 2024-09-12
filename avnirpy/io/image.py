import nibabel as nib
import nrrd
import numpy as np
from typing import List


def axcode_to_ras(axcode: List[str]) -> np.ndarray:
    """
    Converts the axis direction code to the RAS (Right, Anterior, Superior) coordinate system.

    Args:
        axcode (List[str]): A list of three characters representing the axis direction code.

    Returns:
        numpy.ndarray: A diagonal matrix representing the transformation from the axis direction
                       code to the RAS coordinate system.
    """
    xfrm = [1, 1, 1, 1]
    for i, (code, ras) in enumerate(zip(axcode, ["R", "A", "S"])):
        if code != ras:
            xfrm[i] = -1

    return np.diag(xfrm)


def nifti_write(nrrd_image: str, output: str) -> None:
    """
    Write a NIfTI image file from a given input image.

    Parameters:
        nrrd_image (str): The path to the input image file.
        output (str): The path to save the output NIfTI image file.
    """
    img = nrrd.read(nrrd_image)
    hdr = img[1]
    data = img[0]

    translation = hdr["space origin"]
    rotation = hdr["space directions"]
    affine_nhdr = np.vstack(
        (np.hstack((rotation.T, np.reshape(translation, (3, 1)))), [0, 0, 0, 1])
    )
    axcode = nib.orientations.aff2axcodes(affine_nhdr)
    to_ras = axcode_to_ras(axcode)
    affine = np.dot(to_ras, affine_nhdr)

    img_nifti = nib.nifti1.Nifti1Image(data, affine=affine)
    hdr_nifti = img_nifti.header
    hdr_nifti.set_xyzt_units(xyz=2, t=0)
    hdr_nifti["qform_code"] = 1
    hdr_nifti["sform_code"] = 1

    nib.save(img_nifti, output)
