from datetime import datetime
from dataclasses import dataclass

import numpy as np

from .core.weather import foliar_moisture_content, builtup_index, duff_moisture_code, drought_code

@dataclass
class FWIResults:
    fmc: np.ndarray
    dmc_today: np.ndarray
    dc_today: np.ndarray
    bui_today: np.ndarray

class FWIModel:
    def __init__(self,
                 latitude_south: float,
                 latitude_north: float,
                 longitude_east: float,
                 longitude_west: float,
                 shape: tuple[int, int],
                 elevation = None,
                 ) -> None:
        
        self.shape = shape
        h, w = shape
        lat = np.linspace(latitude_north, latitude_south, h)
        lon = np.linspace(longitude_east, longitude_west, w)

        self.lon_arr, self.lat_arr = np.meshgrid(lon, lat)
        self.elevation = elevation
    
    def _to_array(self, attr: np.ndarray | float) -> np.ndarray:
        if isinstance(attr, (int, float)):
            return np.full(self.shape, attr, dtype=type(attr))
        else:
            return attr

    def run(self, date: str | datetime,
            temperature: float | np.ndarray,
            precipitation: float | np.ndarray,
            relative_humidity: float | np.ndarray,
            drought_code_yesterday: float | np.ndarray,
            duff_moisture_code_yesterday: float | np.ndarray
            ) -> FWIResults:
        
        self._temperature = self._to_array(temperature)
        self._precipitation = self._to_array(precipitation)
        self._relative_humidity = self._to_array(relative_humidity)
        self._drought_code_yesterday = self._to_array(drought_code_yesterday)
        self._duff_moisture_code_yesterday = self._to_array(duff_moisture_code_yesterday) 

        
        if isinstance(date, str):
            date = datetime.strptime(date, "%Y-%m-%d")

        doy = date.timetuple().tm_yday
        fmc = foliar_moisture_content(latitude=self.lat_arr,
                                longitude=self.lon_arr,
                                elevation=self.elevation,
                                day_of_year=doy)
        
        dc = drought_code(dc_yesterday=self._drought_code_yesterday,
                          temp=self._temperature,
                          prec=self._precipitation,
                          month=date.month,
                          latitude=self.lat_arr)
        
        dmc = duff_moisture_code(dmc_yesterday=self._duff_moisture_code_yesterday,
                                 temp=self._temperature,
                                 prec=self._precipitation,
                                 rh=self._relative_humidity,
                                 month=date.month,
                                 latitude=self.lat_arr)
        bui = builtup_index(dmc, dc)
        
        results = FWIResults(fmc=fmc,
                             bui_today=bui,
                             dmc_today=dmc,
                             dc_today=dc)
        return results