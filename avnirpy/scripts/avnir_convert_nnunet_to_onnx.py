#!/usr/bin/env python3

"""
Convert a trained nnUNet model to ONNX format.

This script takes a trained nnUNet model directory containing the model checkpoint,
dataset.json, and plans.json files and converts it to ONNX format for deployment.
The converted model will have dynamic input/output dimensions and disabled deep supervision.

Example usage:
    avnir_convert_nnunet_to_onnx /path/to/model/dir output.onnx
"""

import argparse
import torch
from os.path import join
import os

os.environ["nnUNet_raw"] = "/dummy/path"
os.environ["nnUNet_preprocessed"] = "/dummy/path"
os.environ["nnUNet_results"] = "/dummy/path"

from batchgenerators.utilities.file_and_folder_operations import load_json
import nnunetv2
from nnunetv2.inference.predict_from_raw_data import (
    recursive_find_python_class,
)
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
import onnx

from avnirpy.io.utils import (
    assert_inputs_exist,
    add_version_arg,
    add_overwrite_arg,
    assert_outputs_exist,
)


def _build_arg_parser():
    """Build argparser.

    Returns:
        parser (ArgumentParser): Parser built.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input", help="Path to the nnU-Net model file.")
    parser.add_argument("output", help="Path to save the ONNX model file.")

    parser.add_argument(
        "--folds", help="Folds to convert to ONNX format.", default="all"
    )
    add_version_arg(parser)
    add_overwrite_arg(parser)

    return parser


def convert_to_ONNX(
    model_dir: str,
    onnx_model_path: str,
    fold: int = 1,
):
    """
    Convert a trained nnUNet model to ONNX format.

    This function loads a trained nnUNet model from a checkpoint file and converts it to ONNX format
    for deployment. It handles loading the model configuration, weights, and performs the conversion
    with appropriate input/output specifications.

    Args:
        model_dir (str): Directory containing the trained model files including checkpoint,
            dataset.json, and plans.json
        onnx_model_path (str): Output path where the ONNX model will be saved
        fold (int, optional): Which fold to convert if the model was trained with cross-validation.
            Defaults to 1.

    Returns:
        None

    Notes:
        - The function expects the standard nnUNet model directory structure
        - Converts the model with dynamic axes to support variable input sizes
        - Uses ONNX opset version 12
        - Disables deep supervision in the converted model
        - Input tensor format is expected to be (sequence, batch, depth, width, height)
    """
    model_path = "checkpoint_final.pth"
    dataset_json = load_json(join(model_dir, "dataset.json"))
    plans = load_json(join(model_dir, "plans.json"))
    plans_manager = PlansManager(plans)

    parameters = []
    use_folds = [fold]
    for i, f in enumerate(use_folds):
        f = int(f) if f != "all" else f
        checkpoint = torch.load(
            join(model_dir, f"fold_{f}", model_path),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if i == 0:
            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]

        parameters.append(checkpoint["network_weights"])
    configuration_manager = plans_manager.get_configuration(configuration_name)

    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )

    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    model = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,
    )
    for params in parameters:
        model.load_state_dict(params)

    # Export to ONNX
    patch_size = (1, 1, 8, 64, 64)
    dummy = torch.randn(*patch_size)

    torch.onnx.export(
        model.eval(),
        dummy,
        onnx_model_path,
        verbose=False,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "seq", 1: "b", 2: "z", 3: "y", 4: "x"},
            "output": {0: "seq", 1: "b", 2: "z", 3: "y", 4: "x"},
        },
    )
    model = onnx.load(onnx_model_path)
    for inp in model.graph.input:
        shape = str(inp.type.tensor_type.shape.dim)
        print(shape)
    meta = model.metadata_props
    meta.insert(0, onnx.StringStringEntryProto(key="type", value="nnUNet"))
    meta.insert(1, onnx.StringStringEntryProto(key="config", value=configuration_name))
    meta.insert(2, onnx.StringStringEntryProto(key="plans", value=str(plans)))
    meta.insert(3, onnx.StringStringEntryProto(key="dataset", value=str(dataset_json)))
    onnx.save(model, onnx_model_path)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.input, is_directory=True)
    assert_outputs_exist(parser, args, args.output)

    convert_to_ONNX(args.input, args.output, fold=args.folds)


if __name__ == "__main__":
    main()
