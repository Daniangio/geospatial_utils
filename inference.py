import os

import cv2
import onnx
import numpy as np
import onnxruntime as ort
from PIL import Image
import tifffile as tif
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEFAULT_CROP_SHAPE = 512, 512

ONNX_MODEL = os.path.join('onnx_models', 'sen1floods11.onnx')
INPUT_FOLDER = os.path.join('input', 'sava')
OUTPUT_FOLDER = os.path.join('output', 'sava')


def preprocess(image):
    h, w, c = image.shape[:3]
    new_image = np.zeros((h, w, 2), dtype=np.float32)
    image = image.astype(np.float32) / 255
    image = np.clip(np.log10(image) * 10, a_min=-50, a_max=0)
    new_image[:, :, 0] = (image[:, :, 0] - (-17.307974)) / 4.8803816
    new_image[:, :, 1] = ((image[:, :, 1] / 2) - (-10.434534)) / 4.1806455

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(new_image[:, :, 0])
    plt.subplot(1, 2, 2)
    plt.imshow(new_image[:, :, 1])
    plt.show()

    # image = cv2.fastNlMeansDenoisingColored(image, None, 15, 15, 7, 21)
    # image = (image - image.min()) / (image.max() - image.min())
    # image = image[:, :, ::-1].astype(np.float32)
    if h < DEFAULT_CROP_SHAPE[0] or w < DEFAULT_CROP_SHAPE[1]:
        padded_image = np.zeros((max(DEFAULT_CROP_SHAPE[0], h), max(DEFAULT_CROP_SHAPE[1], w), c))
        padded_image += 255
        padded_image[:h, :w, :] = new_image
        return padded_image
    print(np.min(new_image), np.max(new_image))
    return new_image


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 255)
    # vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def get_image_crops(image, crop_shape=DEFAULT_CROP_SHAPE, crop_margin=0.05):
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
        list_of_images.append(batch[i, :, :, :])
    return list_of_images


def get_image_from_crops(image_shape: tuple, image_crops: list, crop_shape=DEFAULT_CROP_SHAPE, crop_margin=0.05):
    h, w = image_shape[:2]
    c = 1 if len(image_crops[0].shape) == 2 else image_crops[0].shape[2]
    image = np.zeros((h, w, c))
    counter = 0
    for y in range(0, h, int(crop_shape[0] * (1 - 2 * crop_margin))):
        condition_on_y = y + crop_shape[0] < h
        y_from = None if y == 0 else (
            y + int(np.ceil(crop_shape[0] * crop_margin)) if condition_on_y else h - int(crop_shape[0] * (1 - crop_margin)))
        y_to = y + int(crop_shape[0] * (1 - crop_margin)) if condition_on_y else None
        for x in range(0, w, int(crop_shape[1] * (1 - 2 * crop_margin))):
            condition_on_x = x + crop_shape[1] < w
            x_from = None if x == 0 else (x + int(np.ceil(crop_shape[1] * crop_margin)) if condition_on_x else w - int(
                crop_shape[1] * (1 - crop_margin)))
            x_to = x + int(crop_shape[1] * (1 - crop_margin)) if condition_on_x else None
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
        image = np.array(Image.open(os.path.join(INPUT_FOLDER, input_filename)))
        h, w = image.shape[:2]

        #new_image = np.zeros((h, w, 2), dtype=np.float32)
        #new_image[:, :, 0] = image[:, :, 0]
        #new_image[:, :, 1] = image[:, :, 1] / 2

        #tif.imwrite('im.tiff', new_image)

        preprocessed_image = preprocess(image)
        image_crops = get_image_crops(preprocessed_image)
        batch = list_to_batch(image_crops)
        print(batch.shape)

        onnx.checker.check_model(model)
        ort_session = ort.InferenceSession(ONNX_MODEL)
        input_names = [x.name for x in ort_session.get_inputs()]
        output = ort_session.run(None, {input_names[0]: batch})

        prediction_crops = batch_to_list(output[0])
        prediction_image = get_image_from_crops(preprocessed_image.shape, prediction_crops)

        prediction_image = prediction_image[:h, :w, :] * 255
        prediction_image = prediction_image.astype(np.uint8)

        plt.imshow(image)
        plt.show()
        plt.imshow(prediction_image)
        plt.show()

        try:
            prediction_image = Image.fromarray(prediction_image.astype(np.uint8))
            prediction_image.save(os.path.join(OUTPUT_FOLDER, output_filename))
        except Exception as e:
            print(f'Failed so save with PIL: {str(e)}')
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, output_filename), prediction_image.astype(np.uint8))


if __name__ == '__main__':
    main()
