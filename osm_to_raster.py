import json
import os
import urllib

import rasterio.mask
from rasterio.features import rasterize

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import cascaded_union

import numpy as np
import matplotlib.pyplot as plt

import requests
from osm2geojson import json2geojson

INPUT_FOLDER = 'output\demv4'
OUTPUT_FOLDER = 'output_no_river\demv4'
RIVER_GEOJSON_FOLDER = 'river_geojson\dem'
RIVER_RASTER_FOLDER = 'river_raster\dem'
OVERPASS = r"https://overpass-api.de/api/interpreter"


def get_river_mask_from_tif(filepath: str, src):
    raster_img = src.read()
    raster_meta = src.meta
    bounds = src.bounds

    bbox = [str(x) for x in bounds]
    inverted_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    bbox_str = ', '.join(inverted_bbox)
    print('BBox:', bbox)

    mydata = f"""
        [out:json];
        (
      // query part for: "water=river"
      way["water"="river"]({bbox_str});
      relation["water"="river"]({bbox_str});
        );
        /*added by auto repair*/
        (._;>;);
        /*end of auto repair*/
        out geom;
        """

    def overpass_call(query):
        encoded = urllib.parse.quote(query.encode('utf-8'), safe='~()*!.\'')
        r = requests.post(OVERPASS,
                          data=f"data={encoded}",
                          headers={'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'})
        if r.status_code != 200:
            print(r.text)
            raise requests.exceptions.HTTPError('Overpass server respond with status ' + str(r.status_code))
        return r.text

    def save_data(data, geom_file):
        json_data = json.dumps(data, indent=2)
        with open(geom_file, 'w') as f:
            f.write(json_data)

    filename = filepath.split('\\')[-1].split('.')[0]
    geojson_path = os.path.join(RIVER_GEOJSON_FOLDER, f'{filename}.json')
    if not os.path.exists(geojson_path):
        data = overpass_call(mydata)
        geojson_data = json2geojson(data)
        save_data(geojson_data, geojson_path)
    train_df = gpd.read_file(geojson_path)

    print(f'CRS Raster: {train_df.crs}, CRS Vector {src.crs}')

    # Generate polygon
    def poly_from_utm(polygon, transform):
        poly_pts = []
        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))

        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly

    # Generate Binary maks
    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        elif row['geometry'].geom_type == 'MultiPolygon':
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    if len(poly_shp) > 0:
        mask = rasterize(shapes=poly_shp,
                         out_shape=im_size)
    else:
        mask = np.zeros(im_size, dtype=np.uint8)
    return mask


def plot_mask(mask):
    # Plot the mask
    plt.figure(figsize=(15, 15))
    plt.imshow(mask)
    plt.show()


def save_mask(mask, src, mask_path):
    # Save the mask
    mask = mask.astype('uint8')
    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1, 'dtype': 'uint8'})
    with rasterio.open(mask_path, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)


if __name__ == '__main__':
    for fn in os.listdir(INPUT_FOLDER):
        if fn.split('.')[-1] not in ['tif', 'tiff']:
            continue
        filepath = os.path.join(INPUT_FOLDER, fn)
        filename = filepath.split('\\')[-1].split('.')[0]
        with rasterio.open(filepath, "r") as src:
            raster_img = src.read()
            raster_img = raster_img.astype('uint8')
            mask_path = os.path.join(RIVER_RASTER_FOLDER, f'{filename}.tif')
            if not os.path.exists(mask_path):
                mask = get_river_mask_from_tif(filepath, src)
                plot_mask(mask)
                save_mask(mask, src, mask_path)
        with rasterio.open(mask_path, "r") as mask:
            mask_img = mask.read()
            src_without_rivers = raster_img - mask_img
            src_without_rivers[src_without_rivers < 128] = 0
            bin_src_without_rivers_meta = src.meta.copy()
            bin_src_without_rivers_meta.update({'count': 1, 'dtype': 'uint8'})
            src_without_rivers_path = os.path.join(OUTPUT_FOLDER, f'{filename}.tif')
            with rasterio.open(src_without_rivers_path, 'w', **bin_src_without_rivers_meta) as dst:
                dst.write(src_without_rivers)
