import numpy as np

from fbp.core.utils import get_fuel_mask


def critical_surface_fire_intensity(fmc, cbh):
    """Eq. 56, FCFDG 1992:
    fmc: folier moisture content
    cbh: crown base height"""
    csi = 0.001 * (cbh ** 1.5) * (460 + 25.9 * fmc) ** 1.5
    return csi

def critical_surface_fire_rate_of_spread(csi, sfc):
    """Eq. 57, FCFDG 1992:
    csi: critical surface fire intensity
    sfc: surface fuel consumption
    """
    rso = csi / (300 * sfc)
    return rso

def crown_fraction_burned(rate_of_spread: np.ndarray,
                          folier_moisture_content: np.ndarray,
                          surface_fuel_consumption: np.ndarray,
                          crown_base_height: np.ndarray | float) -> np.ndarray:

    csi = critical_surface_fire_intensity(folier_moisture_content, crown_base_height)
    rso = critical_surface_fire_rate_of_spread(csi, sfc=surface_fuel_consumption)
    
    """Eq. 58, FCFDG 1992:
    ros: rate of spread
    rso: critical surface fire spread rate"""
    ros = rate_of_spread
    cfb = 1 - np.exp(-0.23 * (ros - rso))
    return cfb

def classify_fire_type(fuel_map: np.ndarray, cfb:np.ndarray | None) -> np.ndarray:
    """Table 15, Hirsch 1196: Type of fire categories
    cfb: crown fraction burned
    S: surface fire
    I: intermittent crown fire
    C: crown fire
    Null: not applicable
    """
    FD = np.full_like(fuel_map, "Null", dtype="<U4")
    

    mask_surface = get_fuel_mask(fuel_map, ["D1", "D2", "O1a", "O1b"])
    if np.any(mask_surface):
        FD[mask_surface] = "S"
    
    mask_crown = ~mask_surface
    if np.any(mask_crown) and cfb is not None:
        FD[mask_crown & (cfb < 0.1)] =  "S"
        FD[mask_crown & (cfb >= 0.1) & (cfb < 0.9)] =  "I"
        FD[mask_crown & (cfb >= 0.9)] =  "C"
    
    return FD
