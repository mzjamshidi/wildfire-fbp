from datetime import datetime
from dataclasses import dataclass

import numpy as np

from .core.weather import foliar_moisture_content

@dataclass
class FWIResults:
    fmc: np.ndarray

class FWI:
    def __init__(self,
                 latitude_south: float,
                 latitude_north: float,
                 longitude_east: float,
                 longitude_west: float,
                 shape: tuple[int, int],
                 elevation = None,
                 ) -> None:
        
        h, w = shape
        lat = np.linspace(latitude_north, latitude_south, h)
        lon = np.linspace(longitude_east, longitude_west, w)

        self.lon_arr, self.lat_arr = np.meshgrid(lon, lat)
        self.elevation = elevation

    def run(self, date: str | datetime):
        
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")

        doy = date.timetuple().tm_yday
        fmc = foliar_moisture_content(latitude=self.lat_arr,
                                longitude=self.lon_arr,
                                elevation=self.elevation,
                                day_of_year=doy)
        
        results = FWIResults(fmc=fmc)
        return results