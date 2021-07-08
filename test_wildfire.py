import gdal
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import tifffile

filename = 'ccc.tiff'

if False:
    img = tifffile.imread(filename)
    plt.imshow(img[:, :, [3, 2, 1]])
    plt.show()
    exit(0)

if False:
    array = np.load('img.npy')
    array = (array * 10).astype(np.uint8)
    print(array.shape, np.min(array), np.max(array))
    ny, nx = array.shape[:2]
    xmin, ymin, xres, yres = [45.0, 7.0, 0.001, 0.001]
    geotransform = (xmin, xres, 0, ymin, 0, -yres)
    # create the N-band raster file
    bands = 1 if len(array.shape) == 2 else array.shape[2]
    dst_ds = gdal.GetDriverByName('GTiff').Create(filename, nx, ny, bands, gdal.GDT_Byte)

    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = gdal.osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(4326)  # 4326 is WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    if bands == 1:
        dst_ds.GetRasterBand(1).WriteArray(array)  # write band to the raster
    else:
        for band in range(bands):
            dst_ds.GetRasterBand(band + 1).WriteArray(array[:, :, band])  # write N-band to the raster
    dst_ds.FlushCache()  # write to disk
    dst_ds = None
    del dst_ds

    exit(0)

ONNX_MODEL = 'onnx_models/concat_unet.onnx'
model = onnx.load(ONNX_MODEL)

image = tifffile.imread(filename)  # np.load('img.npy')
image = image[40:520, 40:520, :12]
h, w = image.shape[:2]

image = (image / 255).astype(np.float32)
# Make image brighter -> network was trained on brighter images (since RGB bands [3, 2, 1] are brightened for
# better visualization
image = image * 2.5
imtoshow = image.copy()
image = (image - 0.5) / 0.5
batch = image.transpose((2, 0, 1)).reshape((1, -1, h, w))

onnx.checker.check_model(model)
ort_session = ort.InferenceSession(ONNX_MODEL)
input_names = [x.name for x in ort_session.get_inputs()]
output = ort_session.run(None, {input_names[0]: batch})


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


bin_out, regr_out = output

bin_out = (np.argmax(bin_out, axis=1) * 255).astype(np.uint8)
image_out = bin_out.transpose((1, 2, 0)).reshape(480, 480)

'''bin_out = np.where(sigmoid(bin_out) >= 0.5, 1.0, 0.0)
image_out = bin_out.squeeze()
image_out = image_out[0, :, :]
image_out = image_out * 255
image_out = image_out.astype(np.uint8)'''

regr_out = regr_out.squeeze(axis=1)
regr_out = (np.rint(np.clip(regr_out, 0, 4)) / 4 * 255).astype(np.uint8)
regr_image_out = regr_out.transpose((1, 2, 0)).reshape(480, 480)

a = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
a[:h, :w] = (imtoshow[:, :, [3, 2, 1]] * 255).astype(np.uint8)
a[h:, :w, 0] = image_out
a[:h, w:, 0] = regr_image_out
plt.imshow(a)
plt.show()
