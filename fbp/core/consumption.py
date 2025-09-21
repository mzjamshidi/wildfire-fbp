import numpy as np

from fbp.constants import FBP_FUEL_MAP
from fbp.core.utils import get_fuel_mask
from fbp.core.crowning import crown_fraction_burned


# Table 8, FCFDG 1992 (unit: meter)
CROWN_BASE_HEIGHT = {   
    "C1": 2,
    "C2": 3,
    "C3": 8,
    "C4": 4,
    "C5": 18,
    "C6": 7,    # NOTE Table 12, Hirch 1996 has a variable value for plantations
    "C7": 10,
    "M1": 6,
    "M2": 6,
    "M3": 6,
    "M4": 6
}

# Table 8, FCFDG 1992 (unit: kg/m^2)
CROWN_FUEL_LOAD = {   
    "C1": 0.75,
    "C2": 0.80,
    "C3": 1.15,
    "C4": 1.20,
    "C5": 1.20,
    "C6": 1.80,
    "C7": 0.50,
    "M1": 0.80,
    "M2": 0.80,
    "M3": 0.80,
    "M4": 0.80
}

CROWNING_FUELS = [f"C{i}" for i in range(1, 8)] + [f"M{i}" for i in range(1, 5)]

def _sfc_c2_formula(bui: np.ndarray):
    """Eq 10, FCFDG 1992"""
    return 5.0 * (1 - np.exp(-0.0115 * bui))

def _sfc_d1_formula(bui: np.ndarray):
    """Eq 16, FCFDG 1992"""
    return  1.5 * (1 - np.exp(-0.0183 * bui))

def _build_cfl(fuel_map: np.ndarray) -> np.ndarray:
    cfl = np.full_like(fuel_map, np.nan, dtype=float)
    for fuel_type, load in CROWN_FUEL_LOAD.items():
        mask = get_fuel_mask(fuel_map, [fuel_type])
        cfl[mask] = load
    return cfl

def surface_fuel_consumption(
        fuel_map: np.ndarray,
        bui: np.ndarray,
        ffmc: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        grass_fuel_load: float = 0.3):

    sfc = np.zeros_like(bui, dtype=float)

    # --- C1 ---
    mask = get_fuel_mask(fuel_map, ["C1"])
    if np.any(mask):
        if ffmc is None:
            raise ValueError(f"ffmc required for C7 (cells: {np.sum(mask)})")
        """Eq. 9a & 9b, Wotton et al. 2009"""
        sfc[mask] = np.where(
            ffmc[mask] > 84,
            0.75 + 0.75 * (1 - np.exp(-0.23 * (ffmc[mask] - 84))) ** 0.5,
            # NOTE in Wotton et al. 2009 the term is written with opposite sign,
            # but the R implementation uses (84 - FFMC) for the FFMC ≤ 84 case
            # which seems to be correct.
            0.75 - 0.75 * (1 - np.exp(-0.23 * (84 - ffmc[mask]))) ** 0.5 
        )
    
    # --- C2, M3 & M4 ---
    mask = get_fuel_mask(fuel_map, ["C2", "M3", "M4"])
    if np.any(mask):
        """Eq 10, FCFDG 1992"""
        sfc[mask] = 5.0 * (1 - np.exp(-0.0115 * bui[mask]))

    # --- C3 & C4 ---
    mask = get_fuel_mask(fuel_map, ["C3", "C4"])
    if np.any(mask):
        """Eq. 11, FCFDG 1992"""
        sfc[mask] = 5.0 * (1 - np.exp(-0.0164 * bui[mask])) ** 2.24
    
    # --- C5 & C6 ---
    mask = get_fuel_mask(fuel_map, ["C5", "C6"])
    if np.any(mask):
        """Eq. 12, FCFDG 1992"""
        sfc[mask] = 5.0 * (1 - np.exp(-0.0149 * bui[mask])) ** 2.48

    # --- C7 ---
    mask = get_fuel_mask(fuel_map, ["C7"])
    if np.any(mask):
        if ffmc is None:
            raise ValueError(f"ffmc required for C7 (cells: {np.sum(mask)})")
        """Eq. 13, FCFDG 1992: forest floor consumption (FFC)"""
        ffc = np.where(
            ffmc[mask] > 70,
            2 * (1 - np.exp(-0.104 * (ffmc[mask] - 70))),
            0)
        """Eq. 14, FCFDG 1992: woody fuel consumption (WFC)"""
        wfc = 1.5 * (1 - np.exp(-0.0201 * bui[mask]))
        """Eq. 15, FCFDG 1992"""
        sfc[mask] = ffc + wfc

    # --- D1 ---
    mask = get_fuel_mask(fuel_map, ["D1"])
    if np.any(mask):
        sfc[mask] = _sfc_d1_formula(bui[mask])

    # --- M1 & M2 ---
    mask = get_fuel_mask(fuel_map, ["M1", "M2"])
    if np.any(mask):
        if percent_conifer_map is None:
            raise ValueError(f"percent_conifer_map required for M1 & M2 (cells: {np.sum(mask)})")
        
        sfc_c2 = _sfc_c2_formula(bui[mask])
        sfc_d1 = _sfc_d1_formula(bui[mask])
        pc = percent_conifer_map[mask]
        """Eq. 17, FCFDG 1992"""
        sfc[mask] = pc / 100 * sfc_c2 + (1 - pc/100) * sfc_d1
    
    # --- O1a & O1b ---
    mask = get_fuel_mask(fuel_map, ["O1a", "O1b"])
    if np.any(mask):
        sfc[mask] = grass_fuel_load
    
    # --- S1 ---
    mask = get_fuel_mask(fuel_map, ["S1"])
    if np.any(mask):
        """Eq. 19, FCFDG 1992: forest floor consumption (FFC)"""
        ffc = 4. * (1 - np.exp(-0.025 * bui[mask]))
        """Eq. 20, FCFDG 1992: woody fuel consumption (WFC)"""
        wfc = 4. * (1 - np.exp(-0.034 * bui[mask]))
        sfc[mask] = ffc + wfc
    
    # --- S2 ---
    mask = get_fuel_mask(fuel_map, ["S2"])
    if np.any(mask):
        """Eq. 21, FCFDG 1992: forest floor consumption (FFC)"""
        ffc = 10. * (1 - np.exp(-0.013 * bui[mask]))
        """Eq. 22, FCFDG 1992: woody fuel consumption (WFC)"""
        wfc = 6. * (1 - np.exp(-0.06 * bui[mask]))
        sfc[mask] = ffc + wfc

    # --- S3 ---
    mask = get_fuel_mask(fuel_map, ["S3"])
    if np.any(mask):
        """Eq. 23, FCFDG 1992: forest floor consumption (FFC)"""
        ffc = 12. * (1 - np.exp(-0.0166 * bui[mask]))
        """Eq. 24, FCFDG 1992: woody fuel consumption (WFC)"""
        wfc = 20. * (1 - np.exp(-0.021 * bui[mask]))
        sfc[mask] = ffc + wfc

    sfc = np.where(sfc <= 0, 1e-6, sfc)
    return sfc

def crown_fuel_consumption(
        fuel_map:np.ndarray,
        cfb: np.ndarray,
        cfl: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None) -> np.ndarray:
    """
    cfb: crown fraction burned
    cfl: crown fuel load
    """
    cfc = np.zeros_like(fuel_map, dtype=float)
    if cfl is None:
        cfl = _build_cfl(fuel_map)
    
    # --- C1 - C7 ---
    mask = get_fuel_mask(fuel_map, [f"C{i}" for i in range(1, 8)])
    if np.any(mask):
        """Eq. 66a, Wotton et al. 2009"""
        cfc[mask] = cfl[mask] * cfb[mask]
    
    # --- M1 & M2 ---
    mask = get_fuel_mask(fuel_map, ["M1", "M2"])
    if np.any(mask):
        if percent_conifer_map is None:
            raise ValueError(f"percent_conifer_map required for M1 & M2 (cells: {np.sum(mask)})")
        pc = percent_conifer_map[mask]
        """Eq. 66b, Wotton et al. 2009"""
        cfc[mask] = cfl[mask] * cfb[mask] * pc / 100.
    
    # -- M3 & M4 ---
    mask = get_fuel_mask(fuel_map, ["M3", "M4"])
    if np.any(mask):
        if percent_dead_fir_map is None:
            raise ValueError(f"percent_dead_fir_map required for M1 & M2 (cells: {np.sum(mask)})")
        pdf = percent_dead_fir_map[mask]
        """Eq. 66c, Wotton et al. 2009"""
        cfc[mask] = cfl[mask] * cfb[mask] * pdf / 100.

    return cfc


def total_fuel_consumption(
        fuel_map: np.ndarray,
        surface_fuel_consumption: np.ndarray,
        crown_fraction_burned: np.ndarray,
        crown_fuel_load: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None
):
    
    tfc = surface_fuel_consumption
    if np.any(get_fuel_mask(fuel_map, CROWNING_FUELS)):
        cfc = crown_fuel_consumption(fuel_map,
                                    crown_fraction_burned,
                                    crown_fuel_load,
                                    percent_conifer_map,
                                    percent_dead_fir_map)
        
        """Eq. 67, FCFDG 1992: total fuel consumption (TFC)"""
        tfc += cfc

    return tfc

def fire_intensity(fc: np.ndarray, ros: np.ndarray) -> np.ndarray:
    """Eq. 69, , FCFDG 1992: fire intensity (FI) (kW/m)
    fc: fuel consumption (surface or total) (kg/m^2)
    ros: rate of spread
    """
    fi = 300 * fc * ros
    return fi