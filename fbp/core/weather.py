import numpy as np

FFMC_COEFFICIENT = 250.0 * 59.5 / 101.0

def fine_fuel_moisture_code(ffmc_yesterday: np.ndarray,
                            temp: np.ndarray,
                            rh: np.ndarray,
                            ws: np.ndarray,
                            prec: np.ndarray) -> np.ndarray:
    
    """Eq. 1, Van Wagner & Pickett 1985"""
    mo = FFMC_COEFFICIENT * (101 - ffmc_yesterday) / (59.5 + ffmc_yesterday)

    """Eq. 1, Van Wagner & Pickett 1985"""
    rf = np.where(prec > 0.5,
                  prec - 0.5,
                  prec)
    
    if np.any(prec > 0.5):    
        """Eq. 3a & 3b, Van Wagner & Pickett 1985: fine fule moisture content after rain (mr)"""
        mr = np.where(mo <= 150,
                    mo + 42.5 * rf * (np.exp(-100 / (251 - mo))) * (1 - np.exp(-6.93 / rf)),
                    mo + 42.5 * rf * (np.exp(-100 / (251 - mo))) * (1 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150) ** 2 * np.sqrt(rf))
        
        # The real moisture content of pine litter ranges up to about 250 percent,
        # so we cap it at 250 (R package)
        mr = np.minimum(mr, 250)
        mo =np.where(prec > 0.5, mr, mo)

    """Eq. 4, Van Wagner & Pickett 1985: Equilibrium moisture content from drying (Ed)"""
    Ed = (0.942 * rh ** 0.679 
          + 11 * np.exp((rh - 100) / 10) 
          + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))
    
    
    """Eq. 6a, Van Wagner & Pickett 1985: Log drying rate at the normal temperature of 21.1C"""
    ko = 0.424 * (1 - (rh / 100) ** 1.7) + 0.0694 * np.sqrt(ws) * (1 - (rh / 100) ** 8)

    """Eq. 6b, Van Wagner & Pickett 1985"""
    kd = ko * 0.581 * np.exp(0.0365 * temp)

    """Eq. 8, Van Wagner & Pickett 1985"""
    md = Ed + (mo - Ed) * 10 ** -kd

    """Eq. 5, Van Wagner & Pickett 1985: Equilibrium moisture content from wetting"""
    Ew = (0.618 * rh ** 0.753
          + 10 * np.exp((rh - 100) / 10) 
          + 0.18 * (21.1 - temp) * (1 - np.exp(-0.115 * rh)))

    """Eq. 7a, Van Wagner & Pickett 1985: Log wetting rate at the normal temperature of 21.1 C"""
    k1 = (0.424 * (1 - ((100 - rh) / 100) ** 1.7) 
          + 0.0694 * np.sqrt(ws) * (1 - ((100 - rh) /100) ** 8))
    
    """Eq. 7b, Van Wagner & Pickett 1985"""
    kw = k1 * 0.581 * np.exp(0.0365 * temp)

    """Eq. 9, Van Wagner & Pickett 1985"""
    mw = Ew - (Ew - mo) * 10 ** -kw

    m = np.where(mo > Ed, md, mw)
    m = np.where((mo <= Ed) & (mo >= Ew), mo, m)

    ffmc_today = 59.5 * (250 - m) / (FFMC_COEFFICIENT + m)
    ffmc_today = np.minimum(ffmc_today, 101.0)
    ffmc_today = np.maximum(ffmc_today, 0)

    return ffmc_today