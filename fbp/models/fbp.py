
from dataclasses import dataclass

import numpy as np

from fbp.core.ros import rate_of_spread, initial_rate_of_spread, buildup_effect
from fbp.core.slope import slope_adjusted_wind_vector
from fbp.core.consumption import total_fuel_consumption, surface_fuel_consumption, fire_intensity
from fbp.core.crowning import crown_fraction_burned, classify_fire_type
from fbp.core.weather import initial_spread_index


@dataclass
class FBPResults:
    fuel: np.ndarray
    cfb: np.ndarray
    # cfc: np.ndarray
    fd: np.ndarray
    hfi: np.ndarray
    wsv: np.ndarray
    raz: np.ndarray
    ros: np.ndarray
    sfc: np.ndarray
    tfc: np.ndarray


class FBPModel:
    def __init__(self,
                 fuel_map: np.ndarray,
                 percent_conifer: np.ndarray | None = None,
                 slope_percent: np.ndarray | float = 0,
                 slope_azimuth: np.ndarray | float = 0) -> None:
        self.fuel_map = fuel_map
        self.percent_conifer = percent_conifer
        self.slope_percent = self._to_array(slope_percent)
        self.slope_azimuth = self._to_array(slope_azimuth)

    def _to_array(self, attr: np.ndarray | float) -> np.ndarray:
        if isinstance(attr, (int, float)):
            return np.full_like(self.fuel_map, attr, dtype=float)
        else:
            return attr

    def run(self,
            fine_fuel_moisture_content: np.ndarray,
            builtup_index: np.ndarray,
            percent_grass_curing: np.ndarray | float | None = None,
            percent_dead_fir: np.ndarray | float | None = None,
            crown_base_height: np.ndarray | float = 2.,
            wind_speed: np.ndarray | float = 0,
            wind_azimuth: np.ndarray | float = 0,
            folier_moisture_content: np.ndarray | float = 0.) -> FBPResults:

        self._ffmc = fine_fuel_moisture_content
        self._bui = builtup_index
        self._fmc = self._to_array(folier_moisture_content)
        self._cbh = crown_base_height

        
        self._percent_dead_fir = self._to_array(percent_dead_fir) if percent_dead_fir is not None else None
        self._percent_grass_curing = self._to_array(percent_grass_curing) if percent_grass_curing is not None else None

        self._wind_speed = self._to_array(wind_speed)
        self._wind_azimuth = self._to_array(wind_azimuth)


        wsv, raz = slope_adjusted_wind_vector(
            fuel_map=self.fuel_map,
            wind_speed=self._wind_speed,
            wind_azimuth=self._wind_azimuth,
            slope_percent=self.slope_percent,
            slope_azimuth=self.slope_azimuth,
            ffmc=self._ffmc,
            percent_conifer_map=self.percent_conifer,
            percent_dead_fir_map=self._percent_dead_fir,
            percent_grass_curing_map=self._percent_grass_curing
        )

        isi = initial_spread_index(ffmc=self._ffmc, ws=wsv)
        rsi = initial_rate_of_spread(self.fuel_map, isi, self._percent_grass_curing, self.percent_conifer)

        be = buildup_effect(self.fuel_map, bui=self._bui)
        ros = rate_of_spread(rsi, be)

        sfc = surface_fuel_consumption(
            fuel_map=self.fuel_map,
            bui=self._bui,
            ffmc=self._ffmc,
            percent_conifer_map=self.percent_conifer
        )

        # TODO this need not to be done if there is not conifer fuel
        cfb = crown_fraction_burned(
            rate_of_spread=ros,
            folier_moisture_content=self._fmc,
            surface_fuel_consumption=sfc,
            crown_base_height=self._cbh
        )

        tfc = total_fuel_consumption(
            fuel_map=self.fuel_map,
            surface_fuel_consumption=sfc,
            crown_fraction_burned=cfb,
            percent_conifer_map=self.percent_conifer
        )

        hfi = fire_intensity(fc=tfc, ros=ros)

        fd = classify_fire_type(fuel_map=self.fuel_map, cfb=cfb)

        results = FBPResults(
            fuel=self.fuel_map,
            ros=ros,
            wsv=wsv,
            raz=raz,
            sfc=sfc,
            cfb=cfb,
            tfc=tfc,
            hfi=hfi,
            fd=fd
        )
        
        return results

