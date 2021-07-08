import json
import os

import cv2
import gdal
import numpy as np
import tifffile
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT_FOLDER = os.path.join('output', 'demv4')
GROUND_TRUTH_FOLDER = os.path.join('ground_truth', 'dem')
VALIDATION_FOLDER = os.path.join('json_validation', 'demv4')


def preprocess_image(image):
    return image


def preprocess_ground_truth(ground_truth):
    new_ground_truth = ground_truth.copy()
    new_ground_truth[new_ground_truth >= 180] = 0
    new_ground_truth[new_ground_truth > 0] = 255
    return new_ground_truth


def main():
    for filename in os.listdir(OUTPUT_FOLDER):
        split = filename.split('.')
        name, extension = split[0], split[1]
        validation_filename = f'{name}.json'
        if os.path.isfile(os.path.join(VALIDATION_FOLDER, validation_filename)):
            continue
        if extension == 'tiff':
            image = np.array(tifffile.imread(os.path.join(OUTPUT_FOLDER, filename)), dtype=np.uint8)
            filename = f'{name}.png'
        elif extension == 'tif':
            image = gdal.Open(os.path.join(OUTPUT_FOLDER, filename)).ReadAsArray()
            image = np.swapaxes(image, 0, 1)
            image = np.swapaxes(image, 1, 2)
            filename = f'{name}.png'
        else:
            image = np.array(Image.open(os.path.join(OUTPUT_FOLDER, filename)))
            if len(image.shape) > 2 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ground_truth = np.array(Image.open(os.path.join(GROUND_TRUTH_FOLDER, filename)))
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)

        print(image.shape)
        plt.imshow(image)
        plt.show()
        plt.imshow(ground_truth)
        plt.show()
        print(image.shape, ground_truth.shape)
        assert image.shape[:2] == ground_truth.shape[:2]

        image = preprocess_image(image)
        ground_truth = preprocess_ground_truth(ground_truth)

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
        print(filename)
        plt.imshow(composed)
        plt.show()

        with open(f'{os.path.join(VALIDATION_FOLDER, validation_filename)}', 'w') as json_file:
            json.dump(validation_dict, json_file)


if __name__ == '__main__':
    main()
