import numpy as np

from fbp import FBP_FUEL_MAP

"""Table 6, FCFDG 1992: Rate of spread parameters for all fuel types (except mixedwood)"""
ROS_PARAMS = {
    "C1" : {"a": 90 , "b": 0.0649, "c": 4.5},
    "C2" : {"a": 110, "b": 0.0282, "c": 1.5},
    "C3" : {"a": 110, "b": 0.0444, "c": 3.0},
    "C4" : {"a": 110, "b": 0.0293, "c": 1.5},
    "C5" : {"a": 30 , "b": 0.0697, "c": 4.0},
    "C6" : {"a": 30 , "b": 0.0800, "c": 3.0},
    "C7" : {"a": 45 , "b": 0.0305, "c": 2.0},
    "D1" : {"a": 30 , "b": 0.0232, "c": 1.6},
    "S1" : {"a": 75 , "b": 0.0297, "c": 1.3},
    "S2" : {"a": 40 , "b": 0.0438, "c": 1.7},
    "S3" : {"a": 55 , "b": 0.0829, "c": 3.2},
    "O1a": {"a": 190, "b": 0.0310, "c": 1.4},
    "O1b": {"a": 250, "b": 0.0350, "c": 1.7},
    }

"""Table 7, FCFDG 1992: Values of BUI0, q, and maximum BE for each fuel type"""
BUILTUP_PARAMS = {
    "C1": {"BUI0": 72  , "q": 0.90, "MaxBE": 1.076},
    "C2": {"BUI0": 64  , "q": 0.70, "MaxBE": 1.321},
    "C3": {"BUI0": 62  , "q": 0.75, "MaxBE": 1.261},
    "C4": {"BUI0": 66  , "q": 0.80, "MaxBE": 1.184},
    "C5": {"BUI0": 56  , "q": 0.80, "MaxBE": 1.220},
    "C6": {"BUI0": 62  , "q": 0.80, "MaxBE": 1.197},
    "C7": {"BUI0": 106 , "q": 0.85, "MaxBE": 1.134},
    "D1": {"BUI0": 32  , "q": 0.90, "MaxBE": 1.179},
    "M1": {"BUI0": 50  , "q": 0.80, "MaxBE": 1.250},
    "M2": {"BUI0": 50  , "q": 0.80, "MaxBE": 1.250},
    "M3": {"BUI0": 50  , "q": 0.80, "MaxBE": 1.250},
    "M4": {"BUI0": 50  , "q": 0.80, "MaxBE": 1.250},
    "S1": {"BUI0": 38  , "q": 0.75, "MaxBE": 1.460},
    "S2": {"BUI0": 63  , "q": 0.75, "MaxBE": 1.256},
    "S3": {"BUI0": 31  , "q": 0.75, "MaxBE": 1.590},
    "O1a": {"BUI0": None, "q": 1.00, "MaxBE": 1.000},
    "O1b": {"BUI0": None, "q": 1.00, "MaxBE": 1.000},
}

def _get_ros_params_m3(pdf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eqs. 29, 30 & 31, FCFDG 1992"""
    a = 170 * np.exp(-35 / pdf)
    b = 0.082 * np.exp(-36 / pdf)
    c = 1.698 - 0.00303 * pdf
    return a, b, c

def _get_ros_params_m4(pdf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eqs. 32, 33 & 34, FCFDG 1992"""
    a = 140 * np.exp(-35.5 / pdf)
    b = 0.0404 * np.ones_like(pdf, dtype=float)
    c = 3.02 * np.exp(-0.00714 * pdf)
    return a, b, c

def _rsi_formula(
        isi: np.ndarray,
        a: float | np.ndarray,
        b: float | np.ndarray,
        c: float | np.ndarray
        ) -> np.ndarray:
    """Eq. 26, FCFDG 1992: Initial rate of spread (RSI)"""
    return a * (1 - np.exp(-b * isi)) ** c


def initial_rate_of_spread(
        fuel_map: np.ndarray,
        isi: np.ndarray,
        percent_grass_curing_map: np.ndarray | None = None,
        percent_conifer_map: np.ndarray | None = None,
        percent_dead_fir_map: np.ndarray | None = None) -> np.ndarray:
        
    rsi = np.full_like(fuel_map, np.nan, dtype=float)

    # --- standard fuels ---
    for fuel, param in ROS_PARAMS.items():
        mask = fuel_map == FBP_FUEL_MAP[fuel]
        if not np.any(mask):
            continue
        
        rsi[mask] = _rsi_formula(isi[mask], **param)

        if fuel in ("O1a", "O1b"):
            if percent_grass_curing_map is None:
                raise ValueError(f"percent_grass_curing_map required for {fuel} (cells: {np.sum(mask)})")
            
            gc = percent_grass_curing_map[mask]
            
            """Eqs. 35(a&b), Wotton et al. 2009 : Revised grass curing coefficient"""
            cf = np.where(
                        gc < 58.8,
                        0.005 * (np.exp(0.061 * gc) - 1),
                        0.176 + 0.02 * (gc - 58.8)
                    )
            
            rsi[mask] *= cf      
    
    # --- mixedwood M1/M2 ---
    for fuel, dec_factor in zip(("M1", "M2"), (1.0, 0.2)):
        mask = fuel_map == FBP_FUEL_MAP[fuel]
        if not np.any(mask):
            continue

        if percent_conifer_map is None:
            raise ValueError(f"percent_conifer_map required for {fuel} (cells: {np.sum(mask)})")
            
        pc = percent_conifer_map[mask]

        rsi_c2 = _rsi_formula(isi[mask], **ROS_PARAMS["C2"])
        rsi_d1 = _rsi_formula(isi[mask], **ROS_PARAMS["D1"])

        """Eqs. 27 & 28, FCFDG 1992"""
        rsi[mask] = (pc / 100) * rsi_c2 + (1 - pc / 100) * rsi_d1 * dec_factor
               
    # --- mixedwood M3 ---
    mask = fuel_map == FBP_FUEL_MAP["M3"]
    if np.any(mask):
        if percent_dead_fir_map is None:
            raise ValueError(f"percent_dead_fir_map required for M3 (cells: {np.sum(mask)})")
        
        pdf = percent_dead_fir_map[mask]
        a, b, c = _get_ros_params_m3(pdf)

        rsi[mask] = _rsi_formula(isi[mask], a, b, c)
    
    # --- mixedwood M4 ---
    mask = fuel_map == FBP_FUEL_MAP["M4"]
    if np.any(mask):
        if percent_dead_fir_map is None:
            raise ValueError(f"percent_dead_fir_map required for M4 (cells: {np.sum(mask)})")
        
        pdf = percent_dead_fir_map[mask]
        a, b, c = _get_ros_params_m4(pdf)

        rsi[mask] = _rsi_formula(isi[mask], a, b, c)

    return rsi


def initial_spread_index(ffmc: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """
        ws: wind speed (km/h)
    """
    wsv = ws    # FIXME this probably not correct.
    FFMC_COEFFICIENT = 250 * 59.5 / 101
    
    """Eq. 46, FCFDG 1992"""
    m = FFMC_COEFFICIENT * (101 - ffmc) / (59.5 + ffmc)

    """Eq. 45, FCFDG 1992"""
    fF = 91.9 * np.exp(-0.1386 * m) * (1 + (m**5.31) / 4.93e7)
    
    """Eqs. 53 & 53a, FCFDG 1992: wsv: net effective wind speed"""
    fW = np.where(
        wsv <= 40,
        np.exp(0.05039 * wsv),
        12 * (1 - np.exp(-0.0818 * (wsv - 28)))
    )

    """Eq. 52, FCFDG 1992"""
    isi = 0.208 * fW * fF

    return isi

def buildup_effect(fuel_map: np.ndarray, bui: np.ndarray) -> np.ndarray:
    be = np.zeros_like(fuel_map, dtype=float)
    for fuel, param in BUILTUP_PARAMS.items():
        mask = fuel_map == FBP_FUEL_MAP[fuel]

        BUI0 = param["BUI0"]
        q = param["q"]

        """Eq. 54, FCFDG 1992: Buildup effect"""
        be[mask] = np.exp(50 * np.log(q) * (1/bui[mask] - 1/BUI0))
    
    return be

def rate_of_spread(rsi: np.ndarray, be) -> np.ndarray:
    """Eq. 55, FCFDG 1992: Rate of spread (ROS)"""
    ros = np.maximum(rsi * be, 1e-6)   
    return ros