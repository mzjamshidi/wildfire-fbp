import numpy as np

from fbp.constants import FBP_FUEL_MAP

def get_fuel_mask(fuel_map: np.ndarray, fuel_types: list[str]) -> np.ndarray:
    fuel_codes = [FBP_FUEL_MAP[f] for f in fuel_types]
    mask = np.isin(fuel_map, fuel_codes)
    return mask
    
