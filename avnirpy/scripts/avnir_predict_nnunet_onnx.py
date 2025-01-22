#!/usr/bin/env python3

"""
Convert a nnU-Net model to ONNX format.
"""

import numpy as np
import os

os.environ["nnUNet_raw"] = "/dummy/path"
os.environ["nnUNet_preprocessed"] = "/dummy/path"
os.environ["nnUNet_results"] = "/dummy/path"

import onnxruntime as ort
import onnx
from onnx.tools import update_model_dims

import argparse
import torch
import nibabel as nib
from os.path import join
from avnirpy.io.utils import assert_inputs_exist, add_version_arg
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
    recursive_find_python_class,
)
from nnunetv2.preprocessing.preprocessors.default_preprocessor import (
    DefaultPreprocessor,
)
import nnunetv2
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Path to the input image.")
    parser.add_argument("model", help="Name of the model configuration.")
    parser.add_argument("output", help="Path to predicted image.")

    add_version_arg(parser)

    return parser


def get_info_from_model(model_path):
    if not model_path.endswith(".onnx"):
        raise ValueError("Model file must be an ONNX file.")

    model = onnx.load(model_path)

    plans = None
    configuration_name = None
    type = None
    dataset = None
    for meta in range(len(model.metadata_props)):
        key = model.metadata_props[meta].key
        value = model.metadata_props[meta].value
        if key == "plans":
            plans = eval(value)
        elif key == "config":
            configuration_name = value
        elif key == "type":
            type = value
        elif key == "dataset":
            dataset = eval(value)

    if type != "nnUNet":
        raise ValueError("Model must be a nnUNet model.")

    input_name = model.graph.input[0].name
    return input_name, dataset, plans, configuration_name


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input)
    input_name, dataset, plans, configuration_name = get_info_from_model(args.model)
    image = nib.load(args.input)
    pp = DefaultPreprocessor(verbose=False)
    plans_manager = PlansManager(plans)
    labels_manager = plans_manager.get_label_manager(dataset)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    # vox_size = image.header.get_zooms()[:3]
    # configuration.configuration["spacing"] = [vox_size[2], vox_size[1], vox_size[0]]
    data, _, data_properties = pp.run_case(
        [args.input],
        seg_file=None,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        dataset_json=dataset,
    )
    preproc = np.expand_dims(data, axis=0)
    print(preproc.shape)
    pad_width = [(0, 0), (0, 0)]
    min_shape = [1, 1, 4, 64, 64]
    for i in range(2, 5):  # Last 3 axes
        if preproc.shape[i] % min_shape[i] == 0:
            pad_width.append((0, 0))
        else:
            next_multiple = ((preproc.shape[i] // min_shape[i]) + 1) * min_shape[i]
            pad_amount = next_multiple - preproc.shape[i]
            pad_width.append((0, pad_amount))
    preproc = np.pad(preproc, pad_width, mode="edge")
    print(preproc.shape)
    sess = ort.InferenceSession(args.model, providers=ort.get_available_providers())
    res = sess.run(None, {input_name: preproc})[0]

    res = unpad(res, pad_width)[0]
    print(res.shape)
    res = convert_predicted_logits_to_segmentation_with_correct_shape(
        res,
        plans_manager,
        configuration_manager,
        labels_manager,
        data_properties,
    )
    res = res.transpose((2, 1, 0))
    print(res.shape)
    res = np.array(res, dtype=np.int8)

    nib.save(nib.Nifti1Image(res, image.affine), args.output)


if __name__ == "__main__":
    main()
