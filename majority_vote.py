import PIL
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from PIL import Image


# ['output/dem/srbac.tiff', 'output/demUnet/srbac.tiff']
# ['output/sava/Srbac 1-37700.tiff', 'output/sava/Srbac 1-37700 GIULIO.tiff', 'output/sava/Srbac 1-37700 SENFLOODS.png']
# ['output/sava/Lijevi Dubrovcak 1-50000.tiff', 'output/sava/Lijevi Dubrovcak 1-50000 GIULIO.tiff', 'output/sava/Lijevi Dubrovcak 1-50000 SENFLOODS.png']

filenames = ['output/sava/Lijevi Dubrovcak 1-50000.tiff', 'output/sava/Lijevi Dubrovcak 1-50000 GIULIO.tiff', 'output/sava/Lijevi Dubrovcak 1-50000 SENFLOODS.png']

images_batch = None
for filename in filenames:
    try:
        image = tifffile.imread(filename)
    except Exception:
        image = np.array(Image.open(filename))
    print(image.shape)
    if images_batch is None:
        images_batch = image
    else:
        images_batch = np.dstack((images_batch, image))

images_batch = np.average(images_batch, axis=-1)
plt.imshow(images_batch)
plt.show()