import json
import os

import cv2
import gdal
import numpy as np
import tifffile
from PIL import Image
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

OUTPUT_FOLDER = os.path.join('output', 'sava')
VALIDATION_FOLDER = os.path.join('retouched', 'sava')


def retouch_image(image):
    kernel = np.ones((7, 7), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


def main():
    for filename in os.listdir(OUTPUT_FOLDER):
        split = filename.split('.')
        name, extension = split[0], split[1]
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        retouched_image = retouch_image(image)

        retouched_image = Image.fromarray(retouched_image)
        retouched_image.save(os.path.join(VALIDATION_FOLDER, filename))


if __name__ == '__main__':
    main()
