"""Script for converting a checkpoint to ONNX format"""


import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import onnx
import onnxruntime
import torch
import typer

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def _load_and_check_onnx_model(onnx_model_path: Path) -> onnx.ModelProto:
    onnx_model = onnx.load(str(onnx_model_path))
    onnx.checker.check_model(onnx_model)

    return onnx_model


def _test_onnx_model(
    onnx_model_path: Path, x: torch.Tensor, torch_out: torch.Tensor
) -> None:
    ort_session = onnxruntime.InferenceSession(str(onnx_model_path))

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def convert_to_onnx(
    checkpoint_path: Path = typer.Option(..., help="Path to checkpoint"),
    onnx_res_path: Path = typer.Option(..., help="Path to result *.onnx model"),
    tensor_size: Tuple[int, int, int] = typer.Option(
        default=(3, 512, 512), help="Number of output model channels"
    ),
    opset_version: int = typer.Option(default=11, help="ONNX opset version"),
) -> None:
    """
    Convert model to ONNX format
    """

    # model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    keyname = 'params'

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    model.eval()

    print("Start torch model inference...")
    x = torch.randn(1, *tensor_size, requires_grad=True)
    torch_out = model(x)
    print("Finished torch model inference...")

    print("Exporting model to ONNX...")
    # Export the model
    torch.onnx.export(
        model=model,  # model being run
        args=x,  # model input (or a tuple for multiple inputs)
        f=str(
            onnx_res_path
        ),  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size", 1: "channels", 2: "x", 3: "y"},  # variable length axes
            "output": {0: "batch_size", 1: "channels", 2: "x", 3: "y"},
        },
    )
    print("Finished exporting model to ONNX...")

    print("Testing model with ONNXRuntime...")
    _load_and_check_onnx_model(onnx_model_path=onnx_res_path)
    _test_onnx_model(onnx_model_path=onnx_res_path, x=x, torch_out=torch_out)
    print("Finished testing model with ONNXRuntime...")


if __name__ == "__main__":
    typer.run(convert_to_onnx)
