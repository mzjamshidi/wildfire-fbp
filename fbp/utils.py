import numpy as np
from rasterio.warp import transform

from fbp.constants import FBP_FUEL_DESC

def make_grid(extent, shape):
    x_min, x_max, y_min, y_max = extent
    h, w = shape
    dx = (x_max - x_min) / w
    dy = (y_max - y_min) / h
    x = np.linspace(x_min + dx/2, x_max - dx/2, w)
    y = np.linspace(y_max - dy/2, y_min + dy/2, h)

    X, Y = np.meshgrid(x, y)
    return X, Y


def to_lat_lon(easting: float, northing: float, src_crs: str):
    lons, lats = transform(src_crs, "EPSG:4326", [easting], [northing])
    return lons[0], lats[0]


def describe_fbp_fuel_types():
    print("="*50)
    for fuel, desc in FBP_FUEL_DESC.items():
        print(f"{fuel:>5} : {desc:<50}")
    print("="*50)