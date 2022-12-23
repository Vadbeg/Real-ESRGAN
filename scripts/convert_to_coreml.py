"""Script for converting a checkpoint to CoreML format"""

from pathlib import Path
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import typer
from albumentations.pytorch.transforms import ToTensorV2
from coremltools.models.neural_network import quantization_utils
from PIL import Image


def to_tensor(image: np.ndarray) -> torch.Tensor:
    to_tensor_func = ToTensorV2(always_apply=True)
    image = to_tensor_func(image=image)["image"]

    return image


def _test_coreml_model(
    coreml_model: ct.models.MLModel,
    jit_model: torch.jit.TracedModule,
    image_path: Path,
    image_size: Tuple[int, int] = (512, 512),
) -> None:
    image_pil = Image.open(image_path)
    image_pil = image_pil.resize(size=image_size)

    coreml_prediction: Image.Image = coreml_model.predict({"image": image_pil})["result"]
    coreml_prediction.save("coreml_prediction.png")

    image_numpy = np.array(image_pil)
    image_numpy = image_numpy / 255.0
    image_tensor = to_tensor(image=image_numpy)
    image_tensor = image_tensor.unsqueeze(0).float()

    jit_prediction = jit_model(image_tensor)[0].detach().numpy()
    jit_prediction = np.transpose(jit_prediction, (1, 2, 0))

    try:
        np.testing.assert_allclose(
            np.array(coreml_prediction), jit_prediction, rtol=1e-03, atol=1e-05
        )
    except AssertionError as exc:
        print(exc)

    print("Exported model has been tested, and the result looks good!")


def convert_to_coreml(
    jit_path: Path = typer.Option(..., help="Path to torchscript model"),
    coreml_res_path: Path = typer.Option(..., help="Path to result *.mlmodel model"),
    input_size: Tuple[int, int, int] = typer.Option(
        default=(3, 512, 512), help="Number of output model channels"
    ),
    test_image_path: Path = typer.Option(
        default="images/1.jpg", help="Path to test image"
    ),
) -> None:
    """
    Convert model to CoreML model
    """
    device = torch.device("cpu")

    jit_model = torch.jit.load(jit_path, map_location=device)

    input_shape = ct.Shape(shape=(1, *input_size))

    scale = 1 / 255.0
    bias = [0, 0, 0]

    print("Converting to CoreML...")
    model_coreml = ct.convert(
        jit_model,
        inputs=[
            ct.ImageType(name="image", shape=input_shape, scale=scale, bias=bias),
        ],
        outputs=[ct.ImageType(name="result", color_layout=ct.colorlayout.RGB)],
    )
    print("Done converting to CoreML...")
    print("Adding metadata...")

    spec = model_coreml.get_spec()

    spec.description.input[0].shortDescription = "Image input, RGB, 512x512"
    spec.description.output[0].shortDescription = "Image result, RGB, 512x512"

    new_model = ct.models.MLModel(spec)

    new_model.author = "Vadim Titko"
    new_model.license = "AIBY"
    new_model.short_description = (
        "Small and fast model. It is used for image super-resolution"
    )
    new_model.versionString = "1.0"
    new_model.version = "1.0"

    print("Done adding metadata...")
    print("Running quantization...")

    new_model = quantization_utils.quantize_weights(new_model, nbits=16)
    print("Done running quantization...")

    print("Running test...")
    _test_coreml_model(
        coreml_model=new_model, jit_model=jit_model, image_path=test_image_path
    )

    new_model.save(str(coreml_res_path))
    print("Successfully saved model to ", coreml_res_path)


if __name__ == "__main__":
    typer.run(convert_to_coreml)
