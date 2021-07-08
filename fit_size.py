import os

import fiona
import rasterio.mask
import rasterio.plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import tifffile
from descartes import PolygonPatch
from rasterio.features import rasterize

'''
srbac.tiff | shape/EMSR358_AOI04_DEL_PRODUCT_observedEventA_r1_v1.shp
samac_0-2.tiff | shape/EMSR358_AOI05_DEL_PRODUCT_observedEventA_r1_v2.shp
prijedor_0-8.tiff | shape/EMSR358_AOI03_DEL_PRODUCT_observedEventA_r1_v1.shp
kutina_0-5.tiff | shape/EMSR275_02KUTINA_DEL_MONIT01_v1_observed_event_a.shp
novska_0-5.tiff | shape/EMSR275_03NOVSKA_DEL_MONIT01_v1_observed_event_a.shp
'''

image_name = "srbac.tiff"
src = rasterio.open(os.path.join('output/dem', image_name))

with fiona.open("shape/EMSR358_AOI04_DEL_PRODUCT_observedEventA_r1_v1.shp", "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

rasterio.plot.show((src, 1))
print(shapes)
out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False)
out_meta = src.meta

# plt.imshow(out_image.transpose(1, 2, 0))
# plt.show()

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(os.path.join('ground_truth/dem', image_name), "w", **out_meta) as dest:
    dest.write(out_image)

image = tifffile.imread(os.path.join('ground_truth/dem', image_name))
plt.imshow(image)
plt.show()
