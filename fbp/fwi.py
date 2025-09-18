from datetime import datetime
from dataclasses import dataclass

import numpy as np

from .core.weather import foliar_moisture_content

@dataclass
class FWIResults:
    fmc: np.ndarray

class FWI:
    def __init__(self,
                 latitude_south,
                 latitude_north,
                 longitude_east,
                 longitude_west,
                 shape,
                 elevation = None,
                 ) -> None:
        
        h, w = shape
        y = np.linspace(latitude_north, latitude_south, h)
        x = np.linspace(longitude_east, longitude_west, w)

        X, Y = np.meshgrid(x, y)
        self.lat_arr = Y
        self.lon_arr = X
        self.elevation = elevation

    def run(self, date: str):
        dt = datetime.strptime(date, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        fmc = foliar_moisture_content(latitude=self.lat_arr,
                                longitude=self.lon_arr,
                                elevation=self.elevation,
                                day_of_year=doy)
        
        results = FWIResults(fmc=fmc)
        return results