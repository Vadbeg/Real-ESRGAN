"""Script for inferencing ONNX model"""

import time
from pathlib import Path

import numpy as np
import onnxruntime
from cv2 import cv2


if __name__ == '__main__':
    model = onnxruntime.InferenceSession('res.onnx')
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name

    image_folder = Path('inputs/')

    image_paths = list(image_folder.glob('*.jpg'))
    image_paths += list(image_folder.glob('*.png'))
    image_paths = ['inputs/img_081_SRF_4_HR_out.png']

    for image_path in image_paths:
        print(f'Processing {image_path}')
        image = cv2.imread(str(image_path))
        # image = cv2.resize(image, (256, 256))

        image_norm = image / 255
        image_norm = image_norm.transpose(2, 0, 1)
        image_norm = image_norm[np.newaxis, ...]
        image_norm = image_norm.astype(np.float32)

        start = time.time()
        ort_inputs = {input_name: image_norm}
        output = model.run(None, ort_inputs)
        output = output[0]
        end = time.time()

        # output = 1 / (1 + np.exp(-output))
        output = output.clip(0, 1)
        output = np.uint8(output * 255)[0]
        output = output.transpose(1, 2, 0)
        image_draw = cv2.resize(image, output.shape[0:2][::-1])

        # print(image_draw.shape)
        # print(output.shape)

        image_draw = cv2.hconcat([image_draw, output])

        print(f'Inference time: {end - start}')
        print(f'Input shape: {image_norm.shape}')
        print(f'Output shape: {output.shape}')

        cv2.imwrite('outputs' + '/' + str(image_path).split('/')[-1], image_draw)
        # cv2.imshow('image', image_draw)
        # cv2.waitKey(0)

