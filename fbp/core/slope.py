import numpy as np

from fbp import FBP_FUEL_MAP
from .ros import (
    initial_rate_of_spread,
    initial_spread_index,
    rate_of_spread,
    _cf_formula,
    _get_ros_params_m3,
    _get_ros_params_m4,
    _fF_formula,
    ROS_PARAMS)

def _slope_factor(gs: np.ndarray) -> np.ndarray:
    """Eq. 39, FCFDG 1992"""
    return np.exp(3.533 * (gs / 100) ** 1.2)

def _isf_basic_formula(
        rsf: np.ndarray,
        a: float | np.ndarray,
        b: float |np.ndarray,
        c: float | np.ndarray) -> np.ndarray:
    """Eq. 41a&b, Wotton 2009"""
    val = 1 - (rsf / a) ** (1/c)
    val = np.clip(val, 0.01, None)
    return np.log(val) / (-b)

def _isf_o1_formula(
        rsf: np.ndarray,
        cf: np.ndarray,
        a: float | np.ndarray,
        b: float |np.ndarray,
        c: float | np.ndarray) -> np.ndarray:
    """Eq. 43a&b, Wotton 2009"""
    val = 1 - (rsf / (a * cf)) ** (1/c)
    val = np.clip(val, 0.01, None)
    return np.log(val) / (-b)

def _wse_formula(isf: np.ndarray, fF: np.ndarray) -> np.ndarray:
    """Eq. 44a,b&c, Wotton 2009: slope equivalent wind speed (WSE)"""
    wse = 1 / 0.05039 * np.log(isf / (0.208 * fF))
    cond1 = (wse > 40) & (isf < 0.999 * 2.496 * fF)
    cond2 = (wse > 40) & (isf >= 0.999 * 2.496 * fF)

    wse = np.where(
        cond1,
        28 - 1 / 0.0818 * np.log(1 - isf / (2.496 * fF)),
        wse
    )
    wse = np.where(cond2, 112.45, wse)
    return wse

def slope_adjusted_zero_wind_rate_of_spread(
        fuel_map: np.ndarray,
        ffmc: np.ndarray,
        percent_ground_slope: np.ndarray,
        percent_grass_curing_map: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None
        ) -> np.ndarray:
    
    # ISI: zero wind on level ground
    isz = initial_spread_index(ffmc, ws=np.zeros_like(ffmc))
    rsi_zero_wind = initial_rate_of_spread(
        fuel_map,
        isi=isz,
        percent_conifer_map=percent_conifer_map,
        percent_dead_fir_map=percent_dead_fir_map,
        percent_grass_curing_map=percent_grass_curing_map)

    # RSZ: zero wind rate of spread (no built-up)
    rsz = rate_of_spread(rsi_zero_wind,
                         be=np.full_like(ffmc, 1.0))

    sf = _slope_factor(percent_ground_slope)

    """Eq. 40, FCFDG 1992: slope-adjusted zero wind rate of spread (RSF)"""
    rsf  = rsz * sf
    return rsf

def slope_adjusted_initial_spread_index(
        fuel_map: np.ndarray,
        rsf: np.ndarray,
        percent_grass_curing_map: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None) -> np.ndarray:
    
    isf = np.full_like(rsf, np.nan, dtype=float)
    
    # --- standard fuels --   
    for fuel, params in ROS_PARAMS.items():
        mask = fuel_map == FBP_FUEL_MAP[fuel]
        if not np.any(mask):
            continue

        if fuel in ["O1a", "O1b"]:
            if percent_grass_curing_map is None:
                raise ValueError(f"percent_grass_curing_map required for {fuel} (cells: {np.sum(mask)})")
            gc = percent_grass_curing_map[mask]
            cf = _cf_formula(gc)
            isf[mask] = _isf_o1_formula(rsf[mask], cf, **params)
        
        else:
            params = ROS_PARAMS[fuel]
            isf[mask] = _isf_basic_formula(rsf[mask], **params)

    # --- mixedwood M1/M2 ---
    for fuel in ["M1", "M2"]:
        mask = fuel_map == FBP_FUEL_MAP[fuel]
        if not np.any(mask):
            continue

        if percent_conifer_map is None:
            raise ValueError(f"percent_conifer_map required for {fuel} (cells: {np.sum(mask)})")
        
        pc = percent_conifer_map[mask]
        isf_c2 =  _isf_basic_formula(rsf[mask], **ROS_PARAMS["C2"])
        isf_d1 = _isf_basic_formula(rsf[mask], **ROS_PARAMS["D1"])

        """Eq. 42a, Wotton 2009"""
        isf[mask] = pc/100 * isf_c2 + (1 - pc/100) * isf_d1

    # --- mixedwood M3/M4 ---
    for fuel in ["M3", "M4"]:
        mask = fuel_map == FBP_FUEL_MAP[fuel]
        if np.any(mask):
            if percent_dead_fir_map is None:
                raise ValueError(f"percent_dead_fir_map required for {fuel} (cells: {np.sum(mask)})")
            
            pdf100 = np.full_like(mask, 100, dtype=float)
            if fuel == "M3":
                a, b, c = _get_ros_params_m3(pdf100[mask])

            if fuel == "M4":
                a, b, c = _get_ros_params_m4(pdf100[mask])
            
            isf_m100 = _isf_basic_formula(rsf[mask], a, b, c)
            isf_d1 = _isf_basic_formula(rsf[mask], **ROS_PARAMS["D1"])

            pdf = percent_dead_fir_map[mask]

            """Eq. 42b&c, Wotton 2009"""
            isf[mask] = (pdf/100) * isf_m100 + (1 - pdf/100) * isf_d1
    
    return isf

def slope_adjusted_wind_vector(
        fuel_map: np.ndarray,
        wind_speed: np.ndarray,
        wind_azimuth: np.ndarray,
        slope_percent: np.ndarray,
        slope_azimuth: np.ndarray,
        ffmc: np.ndarray,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None,
        percent_grass_curing_map: np.ndarray | None = None
        ) -> tuple[np.ndarray, np.ndarray]: 
    
    ws = wind_speed
    waz = wind_azimuth * np.pi / 180
    saz = slope_azimuth * np.pi / 180

    rsf = slope_adjusted_zero_wind_rate_of_spread(
        fuel_map=fuel_map,
        ffmc=ffmc,
        percent_ground_slope=slope_percent,
        percent_conifer_map=percent_conifer_map,
        percent_dead_fir_map=percent_dead_fir_map,
        percent_grass_curing_map=percent_grass_curing_map)

    isf = slope_adjusted_initial_spread_index(
        rsf=rsf,
        fuel_map=fuel_map,
        percent_conifer_map=percent_conifer_map,
        percent_dead_fir_map=percent_dead_fir_map,
        percent_grass_curing_map=percent_grass_curing_map
        )

    fF = _fF_formula(ffmc)
    wse = _wse_formula(isf, fF)

    """Eq. 47 & 48, Wotton 2009"""
    wsx = ws * np.sin(waz) + wse * np.sin(saz)
    wsy = ws * np.cos(waz) + wse * np.cos(saz)

    """Eq. 49 & 50, Wotton 2009"""
    wsv = np.sqrt(wsx**2 + wsy**2)
    raz = np.arccos(wsy / wsv) * 180 / np.pi
    
    """Eq. 51, Wotton 2009"""
    raz = np.where(wsx < 0, 360 - raz, raz)

    return wsv, raz


