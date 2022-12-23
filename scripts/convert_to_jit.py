"""Script for converting a checkpoint to TorchScript format"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import typer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


class SuperResCoreMLModule(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self._model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model(x)

        x = torch.clip(input=x, min=0.0, max=1.0)
        x = x * 255.0

        return x


def _test_torchscript_model(
    jit_model: torch.jit.TracedModule, x: torch.Tensor, torch_out: torch.Tensor
) -> None:
    jit_out = jit_model(x)
    np.testing.assert_allclose(
        to_numpy(torch_out), to_numpy(jit_out), rtol=1e-03, atol=1e-05
    )

    print("Exported model has been tested with jit runtime, and the result looks good!")


def convert_to_jit(
    checkpoint_path: Path = typer.Option(..., help="Path to checkpoint"),
    jit_res_path: Path = typer.Option(..., help="Path to result *.jit.pt model"),
    tensor_size: Tuple[int, int, int] = typer.Option(
        default=(3, 512, 512), help="Number of output model channels"
    ),
) -> None:
    """
    Convert model to JIT torchscript format
    """

    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    keyname = 'params'
    if 'params_ema' in checkpoint:
        keyname = 'params_ema'

    model.load_state_dict(checkpoint[keyname])
    # set the train mode to false since we will only run the forward pass.
    model.train(False)
    model.cpu().eval()

    model = SuperResCoreMLModule(model)
    model.eval()

    x = torch.randn(1, *tensor_size, requires_grad=True)

    print('Running model inference...')
    torch_out = model(x)

    print("Exporting model to jit format")
    # Export the model
    jit_model = torch.jit.trace(model, example_inputs=x)
    _test_torchscript_model(jit_model=jit_model, x=x, torch_out=torch_out)

    jit_model.save(str(jit_res_path))


if __name__ == "__main__":
    typer.run(convert_to_jit)
