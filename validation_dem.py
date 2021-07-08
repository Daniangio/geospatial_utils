import json
import os

import cv2
import gdal
import numpy as np
import tifffile
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT_FOLDER = os.path.join('output_no_river', 'demv4')
IMAGE_FOLDER = os.path.join('images', 'demv4_no_river')
GROUND_TRUTH_FOLDER = os.path.join('ground_truth', 'dem')
VALIDATION_FOLDER = os.path.join('json_validation', 'demv4_no_river')


def preprocess_image(image):
    return image


def preprocess_ground_truth(ground_truth):
    return ground_truth.astype(np.uint8)


def main():
    for filename in os.listdir(OUTPUT_FOLDER):
        print(filename)
        split = filename.split('.')
        name, extension = split[0], split[1]
        validation_filename = f'{name}.json'
        if os.path.isfile(os.path.join(VALIDATION_FOLDER, validation_filename)):
            continue
        if extension == 'tiff':
            image = np.array(tifffile.imread(os.path.join(OUTPUT_FOLDER, filename)), dtype=np.uint8)
            filename = f'{name}.tiff'
        elif extension == 'tif':
            image = gdal.Open(os.path.join(OUTPUT_FOLDER, filename)).ReadAsArray()
            if len(image.shape) == 3:
                image = np.swapaxes(image, 0, 1)
                image = np.swapaxes(image, 1, 2)
            filename = f'{name}.tiff'
        else:
            image = np.array(Image.open(os.path.join(OUTPUT_FOLDER, filename)))
            if len(image.shape) > 2 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            ground_truth = tifffile.imread(os.path.join(GROUND_TRUTH_FOLDER, filename))
        except Exception:
            print(f'File {os.path.join(GROUND_TRUTH_FOLDER, filename)} missing. Cannot validate this area')
            continue

        assert image.shape[:2] == ground_truth.shape[:2]

        image = preprocess_image(image)
        ground_truth = preprocess_ground_truth(ground_truth)

        # print(image.dtype, ground_truth.dtype)
        intersection = cv2.bitwise_and(image, ground_truth)
        precision = np.sum(intersection) / np.sum(image)
        recall = np.sum(intersection) / np.sum(ground_truth)
        f_score = 2 * precision * recall / (precision + recall)
        validation_dict = {
            'ground_truth_acquisition_date': '',
            'prediction_acquisition_date': '',
            'precision': precision,
            'recall': recall,
            'f-score': f_score
        }

        composed = np.vstack((ground_truth, image, intersection))

        from PIL import Image
        im = Image.fromarray(composed)
        im.save(os.path.join(IMAGE_FOLDER, f"{name}.jpeg"))

        print(filename)
        #plt.imshow(composed)
        #plt.show()

        with open(f'{os.path.join(VALIDATION_FOLDER, validation_filename)}', 'w') as json_file:
            json.dump(validation_dict, json_file)


if __name__ == '__main__':
    main()
