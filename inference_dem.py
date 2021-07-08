import math
import os
from io import BytesIO

import cv2
import onnx
import numpy as np
import onnxruntime as ort
import tifffile
from PIL import Image
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEFAULT_CROP_SHAPE = 512, 512

ONNX_MODEL = os.path.join('onnx_models', 'DL_norm.onnx')
INPUT_FOLDER = os.path.join('input', 'dem')
OUTPUT_FOLDER = os.path.join('output', 'dem')


def preprocess(image):
    h, w, c = image.shape

    R = np.nan_to_num(np.array(image[:, :, 0], dtype=np.float32), nan=0)
    R = np.clip(np.vectorize(lambda x: 10 * math.log(x + 1e-8))(R), a_min=-50, a_max=0)
    R = np.nan_to_num(np.vectorize(lambda x: (x + 50) / (50))(R), nan=0)
    R = (R - 0.43877342002166553) / 0.21975871894472046

    G = np.nan_to_num(
        np.array(image[:, :, 1], dtype=np.float32), nan=0)
    G = np.clip(np.vectorize(lambda x: 10 * math.log(x + 1e-8))(G), a_min=-50, a_max=0)
    G = np.nan_to_num(np.vectorize(lambda x: (x + 50) / (50))(G), nan=0)
    G = (G - 0.20594363912372493) / 0.14778862117920671

    B = np.nan_to_num(np.array(image[:, :, 2], dtype=np.float32), nan=0)
    B[B < -50] = -50
    B = np.vectorize(lambda x: (x + 50) / 1550)(B)
    B = (B - 0.03255113922968425) / 0.00454465936184611

    # Put all together
    image = np.asarray(np.array([R, G, B]).transpose(1, 2, 0), dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, 0])
    plt.subplot(1, 3, 2)
    plt.imshow(image[:, :, 1])
    plt.subplot(1, 3, 3)
    plt.imshow(image[:, :, 2])
    plt.show()

    if h < DEFAULT_CROP_SHAPE[0] or w < DEFAULT_CROP_SHAPE[1]:
        padded_image = np.zeros((max(DEFAULT_CROP_SHAPE[0], h), max(DEFAULT_CROP_SHAPE[1], w), c))
        padded_image += 255
        padded_image[:h, :w, :] = image
        return padded_image
    return image


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 255)
    # vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def get_image_crops(image, crop_shape=DEFAULT_CROP_SHAPE, crop_margin=0):
    image_crops = []
    h, w, c = image.shape[:3]
    for y in range(0, h, int(crop_shape[0] * (1 - 2 * crop_margin))):
        condition_on_y = y + crop_shape[0] < h
        y_from = y if condition_on_y else h - crop_shape[0]
        y_to = y + crop_shape[0] if condition_on_y else None
        for x in range(0, w, int(crop_shape[1] * (1 - 2 * crop_margin))):
            condition_on_x = x + crop_shape[1] < w
            x_from = x if condition_on_x else w - crop_shape[1]
            x_to = x + crop_shape[1] if condition_on_x else None
            image_crop = np.array(image[y_from:y_to, x_from:x_to, :])
            image_crops.append(image_crop)
    return image_crops


def list_to_batch(list_of_images):
    h, w, c = list_of_images[0].shape
    b = len(list_of_images)
    batch = np.zeros((b, h, w, c), dtype=np.float32)
    for i, image in enumerate(list_of_images):
        batch[i, :, :, :] = image
    return batch


def batch_to_list(batch):
    list_of_images = []
    for i in range(batch.shape[0]):
        list_of_images.append(batch[i, :, :, :].transpose(1, 2, 0))
    return list_of_images


def get_image_from_crops(image_shape: tuple, image_crops: list, crop_shape=DEFAULT_CROP_SHAPE, crop_margin=0):
    h, w = image_shape[:2]
    c = 1 if len(image_crops[0].shape) == 2 else image_crops[0].shape[2]
    image = np.zeros((h, w, c))
    counter = 0
    for y in range(0, h, int(crop_shape[0] * (1 - 2 * crop_margin))):
        condition_on_y = y + crop_shape[0] < h
        y_from = None if y == 0 else (
            y + int(np.ceil(crop_shape[0] * crop_margin)) if condition_on_y else h - int(
                crop_shape[0] * (1 - crop_margin)))
        y_to = y + int(crop_shape[0] * (1 - crop_margin)) if condition_on_y else None
        for x in range(0, w, int(crop_shape[1] * (1 - 2 * crop_margin))):
            condition_on_x = x + crop_shape[1] < w
            x_from = None if x == 0 else (x + int(np.ceil(crop_shape[1] * crop_margin)) if condition_on_x else w - int(
                crop_shape[1] * (1 - crop_margin)))
            x_to = x + int(crop_shape[1] * (1 - crop_margin)) if condition_on_x else None
            print(image.shape, image_crops[0].shape)
            print(y_from, y_to, x_from, x_to, len(image_crops), counter)
            image[y_from:y_to, x_from:x_to, ...] = image_crops[counter][None if y_from is None else int(np.ceil(
                crop_shape[0] * crop_margin)):None if y_to is None else int(crop_shape[0] * (1 - crop_margin)),
                                                   None if x_from is None else int(np.ceil(
                                                       crop_shape[1] * crop_margin)):None if x_to is None else int(
                                                       crop_shape[1] * (1 - crop_margin))]
            counter += 1
    return image


def main():
    model = onnx.load(ONNX_MODEL)

    for input_filename in os.listdir(INPUT_FOLDER):
        output_filename = f'{input_filename.split(".")[0]}.png'
        image = tifffile.imread(os.path.join(INPUT_FOLDER, input_filename))
        h, w = image.shape[:2]

        preprocessed_image = preprocess(image)
        print(preprocessed_image.shape)
        image_crops = get_image_crops(preprocessed_image)
        batch = list_to_batch(image_crops)
        print(batch.shape)
        batch = batch.transpose(0, 3, 1, 2)
        batch = np.concatenate((batch, batch), axis=0)
        print(batch.shape)

        onnx.checker.check_model(model)
        ort_session = ort.InferenceSession(ONNX_MODEL)
        input_names = [x.name for x in ort_session.get_inputs()]
        output = ort_session.run(None, {input_names[0]: batch})
        print(output[0].shape)
        #prediction_crops = batch_to_list(output[0])
        #prediction_image = get_image_from_crops(preprocessed_image.shape, prediction_crops)

        #prediction_image = prediction_image[:h, :w, :]
        prediction_image = output[0][0, :, :, :].transpose(1, 2, 0)
        print(prediction_image.shape)
        prediction_image_final = (np.argmax(prediction_image, axis=2) * 255).astype(np.uint8)
        print(prediction_image_final.shape)
        buf = BytesIO()
        data = Image.fromarray(prediction_image_final)
        data.save(buf, 'png')
        plt.imshow(prediction_image_final)
        plt.show()
        #plt.imshow(prediction_image[:, :, 0])
        #plt.show()
        #plt.imshow(prediction_image[:, :, 1])
        #plt.show()

        try:
            prediction_image = Image.fromarray(prediction_image.astype(np.uint8))
            prediction_image.save(os.path.join(OUTPUT_FOLDER, output_filename))
        except Exception as e:
            print(f'Failed so save with PIL: {str(e)}')
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_filename), prediction_image.astype(np.uint8))


if __name__ == '__main__':
    main()
