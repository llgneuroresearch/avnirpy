import os

from PIL import Image, ImageDraw, ImageFont
import nibabel as nib
import numpy as np

colors = {
    "blue": (0.0, 0.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "purple": (0.6275, 0.1255, 0.9412),
    "cyan": (0.0, 1.0, 1.0),
    "green": (0.0, 1.0, 0.0),
    "orange": (1.0, 0.5, 0.0),
    "white": (1.0, 1.0, 1.0),
    "brown": (0.5, 0.1647, 0.1647),
    "grey": (0.7529, 0.7529, 0.7529),
}


def screenshot_mosaic_wrapper(
    filename,
    output_prefix="",
    directory=".",
    skip=1,
    pad=20,
    nb_columns=15,
    return_path=True,
    min_val=None,
    max_val=None,
    offset_percent=0,
    is_labels=False,
):
    """
    Compute mosaic wrapper from an image

    Parameters
    ----------
    filename : string
        Image filename.
    output_prefix : string
        Image_prefix.
    directory : string
        Directory to save the mosaic.
    skip : int
        Number of images to skip between 2 images in the mosaic.
    pad : int
        Padding value between each images.
    nb_columns : int
        Number of columns.
    return_path : bool
        Return path of the mosaic.
    min_val : float
        Minimum value for the colormap.
    max_val : float
        Maximum value for the colormap.
    offset_percent : float
        Offset percentage for the mosaic.

    Returns
    -------
    name : string
        Path of the mosaic
    imgs_comb : array 2D
        mosaic in array 2D
    """
    data = nib.load(filename).get_fdata()
    offset = round(data.shape[2] * offset_percent)
    skip = round((data.shape[2] - 2 * offset) / nb_columns)
    data = np.nan_to_num(data)

    output_prefix = output_prefix.replace(" ", "_") + "_"

    if is_labels:
        lut = {}
        unique = np.unique(data.astype(np.int8))
        lut[0] = np.array((0, 0, 0), dtype=np.int8)
        for curr_label in unique[1:]:
            color = list(colors.values())[curr_label]
            lut[curr_label] = np.array(
                (
                    color[0] * 255,
                    color[1] * 255,
                    color[2] * 255,
                ),
                dtype=np.int8,
            )
        tmp = np.zeros(data.shape + (3,))
        for label in unique:
            tmp[data == label] = lut[label]
        data = tmp

    imgs_comb = screenshot_mosaic(data, skip, pad, nb_columns, min_val, max_val, offset)
    if return_path:
        image_name = os.path.basename(str(filename)).split(".")[0]
        name = os.path.join(directory, output_prefix + image_name + ".png")
        imgs_comb.save(name)
        return name
    else:
        return imgs_comb


def screenshot_mosaic_blend(
    image,
    image_blend,
    output_prefix="",
    directory=".",
    blend_val=0.5,
    skip=1,
    pad=20,
    nb_columns=15,
    min_val=None,
    max_val=None,
    offset_percent=0,
):
    mosaic_image = screenshot_mosaic_wrapper(
        image,
        skip=skip,
        pad=pad,
        nb_columns=nb_columns,
        return_path=False,
        min_val=min_val,
        max_val=max_val,
        offset_percent=offset_percent,
    )
    mosaic_blend = screenshot_mosaic_wrapper(
        image_blend,
        skip=skip,
        pad=pad,
        nb_columns=nb_columns,
        return_path=False,
        offset_percent=offset_percent,
        is_labels=True,
    )

    output_prefix = output_prefix.replace(" ", "_") + "_"
    image_name = os.path.basename(str(image)).split(".")[0]
    blend = Image.blend(mosaic_image, mosaic_blend, alpha=blend_val)
    name = os.path.join(directory, output_prefix + image_name + ".png")
    blend.save(name)
    return name


def screenshot_mosaic(
    data, skip, pad, nb_columns, min_val=None, max_val=None, offset=0
):
    """
    Compute a mosaic from an image

    Parameters
    ----------
    data : array 3D or 4D
        Data for the mosaic.
    skip : int
        Number of images to skip between 2 images in the mosaic.
    pad : int
        Padding value between each images.
    nb_columns : int
        Number of columns.
    axis : bool
        Display axis.
    min_val : float
        Minimum value for the colormap.
    max_val : float
        Maximum value for the colormap.
    offset : int
        Offset index for the mosaic.


    Returns
    -------
    gif : array 3D
        GIF in array 3D
    imgs_comb : array 2D
        mosaic in array 2D
    """
    range_row = range(0, data.shape[2] - 2 * offset, skip)
    nb_rows = int(np.ceil(len(range_row) / nb_columns))
    shape = (
        (data[:, :, 0].shape[1] + pad) * nb_rows + pad * nb_rows,
        (data[:, :, 0].shape[0] + pad) * nb_columns + nb_columns * pad,
    )
    padding = ((int(pad / 2), int(pad / 2)), (int(pad / 2), int(pad / 2)))
    axis_padding = ((50, 50), (50, 50))
    if data.ndim < 4:
        if min_val is None:
            min_val = np.min(data[data > 0])
        if max_val is None:
            max_val = np.percentile(data[data > 0], 99)
        if max_val - min_val < 20 and max_val.is_integer():
            min_val = data.min()
            max_val = np.percentile(data[data > 0], 99.99)
        data = np.interp(data, xp=[min_val, max_val], fp=[0, 255]).astype(
            dtype=np.uint8
        )
    else:
        # Image is RGB
        shape += (3,)
        padding += ((0, 0),)
        axis_padding += ((0, 0),)

    mosaic = np.zeros(shape, dtype=np.uint8)

    for i, idx in enumerate(range_row):
        corner = i % nb_columns
        row = int(i / nb_columns)
        curr_img = np.rot90(data[:, :, idx + offset])
        curr_img = np.pad(curr_img, padding, "constant").astype(dtype=np.uint8)
        curr_shape = curr_img.shape
        mosaic[
            curr_shape[0] * row
            + row * pad : row * curr_shape[0]
            + curr_shape[0]
            + row * pad,
            curr_shape[1] * corner
            + corner * pad : corner * curr_shape[1]
            + curr_shape[1]
            + corner * pad,
        ] = curr_img

    mosaic = np.pad(mosaic, axis_padding, "constant").astype(dtype=np.uint8)
    if data.ndim < 4:
        img = Image.fromarray(mosaic)
        draw = ImageDraw.Draw(img)
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 40)
        draw.text([mosaic.shape[1] / 2, 0], "A", fill=255, font=fnt)
        draw.text([mosaic.shape[1] / 2, mosaic.shape[0] - 40], "P", fill=255, font=fnt)
        draw.text([0, mosaic.shape[0] / 2], "L", fill=255, font=fnt)
        draw.text([mosaic.shape[1] - 40, mosaic.shape[0] / 2], "R", fill=255, font=fnt)
        mosaic = np.array(img, dtype=np.uint8)

    img = np.uint8(np.clip(mosaic, 0, 255))
    imgs_comb = Image.fromarray(img)
    if mosaic[:, :].shape[1] > 1920:
        basewidth = 1920
        wpercent = basewidth / float(imgs_comb.size[0])
        hsize = int((float(imgs_comb.size[1]) * float(wpercent)))
        imgs_comb = imgs_comb.resize((basewidth, hsize), Image.LANCZOS)
    imgs_comb = imgs_comb.convert("RGB")
    return imgs_comb
