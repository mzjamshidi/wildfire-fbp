from typing import Any, Dict, Union

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling, calculate_default_transform

class Layer:
    def __init__(self, data: np.ndarray, meta: dict) -> None:
        
        self._data = data.copy()
        self.meta = meta.copy()

    @property
    def data(self):
        if self._data.ndim == 3 and self._data.shape[0] == 1:
            return self._data[0]
        return self._data
    
    @property
    def count(self) -> int:
        return self.meta.get("count", self._data.shape[0] if self._data.ndim == 3 else 1)
    
    @property
    def extent(self):
        bounds = self.meta.get("bounds")
        if bounds:
            return (bounds.left, bounds.right, bounds.bottom, bounds.top)
        return None 
    
    @property
    def shape(self):
        return self.data.shape


    def reproject(self, dst_crs, method):
        method_map = {
            "bilinear": Resampling.bilinear,
            "nearest": Resampling.nearest
        }

        resampling_method = method_map[method]

        data = self.data
        meta = self.meta
        count = meta["count"]
        height, width = meta["height"], meta["width"]
        src_crs = meta["crs"] 
        src_transform = meta["transform"]
        bounds = meta["bounds"]

        if src_crs == dst_crs:
            print("Source and destination CRS are the same. No reprojection needed.")
            return

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=width,
            height=height,
            left=bounds.left,
            bottom=bounds.bottom,
            right=bounds.right,
            top=bounds.top
        )
        
        assert dst_height and dst_width
        reprojected = np.empty(shape=(count, int(dst_height), int(dst_width)), dtype=meta["dtype"])

        reproject(
            source=data,
            destination=reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling_method
        )
        meta.update({
            "crs": dst_crs,
            "transform": dst_transform,
            "width": dst_width,
            "height": dst_height,
            "bounds": BoundingBox(left=dst_transform[2],
                    bottom=dst_transform[5] + dst_height * dst_transform[4],
                    right=dst_transform[2] + dst_width * dst_transform[0],
                    top=dst_transform[5])
            })
        self._data = reprojected
        print(f"Layer reprojected from {src_crs} to {dst_crs}.")

    def resize(self, shape, method):
        method_map = {
            "bilinear": Resampling.bilinear,
            "nearest": Resampling.nearest
        }

        resampling_method = method_map[method]
        dst_height, dst_width = shape

        data = self.data
        meta = self.meta
        count = meta["count"]
        src_height, src_width = meta["height"], meta["width"]
        src_crs = meta["crs"] 
        src_transform = meta["transform"]
        bounds = meta["bounds"]


        if (src_height == dst_height) and (src_width == dst_width):
            print("Source and destination have the same size. No resizing needed.")
            return

        dst_transform = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, dst_width, dst_height
            )

        resized = np.empty(shape=(count, dst_height, dst_width), dtype=self.meta["dtype"])
        reproject(
            source=data,
            destination=resized,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=src_crs,
            resampling=resampling_method
        )

        self._data = resized
        self.meta.update({
        "transform": dst_transform,
        "width": dst_width,
        "height": dst_height,
        "bounds": bounds
        })

        print(f"Layer resized from ({src_height}, {src_width}) to ({dst_height}, {dst_width})")
        
    def save(self):
        raise NotImplemented

    def __call__(self) -> np.ndarray:
        return self.data.copy()
    
    def __str__(self):
        return f"Layer(shape={self.data.shape}, dtype={self.data.dtype}, crs={self.meta.get('crs', 'unknown')})"
    
class GeoTiffLayer(Layer):
    def __init__(self, path: str) -> None:

        with rasterio.open(path) as src:
            data = src.read()
            meta = src.meta.copy()
            bounds = src.bounds
        
        meta["bounds"] = bounds

        # self.extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        
        super().__init__(data, meta)

class ChildLayer(Layer):
    def __init__(self, data: np.ndarray, parent: Layer) -> None:
        meta = parent.meta.copy()
        bounds = meta["bounds"]
        
        if data.ndim == 2:
            count, height, width = 1, *data.shape
        else:
            count, height, width = data.shape

        transform = from_bounds(
            west=bounds.left, south=bounds.bottom, east=bounds.right, north=bounds.top,
            width=width, height=height
        )

        meta.update({
            "transform": transform,
            "dtype": data.dtype,
            "count": count,
            "height": height,
            "width": width,
        })

        super().__init__(data=data, meta=meta)
        