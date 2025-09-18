from datetime import datetime

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

def foliar_moisture_content(latitude: np.ndarray,
                            longitude: np.ndarray,
                            day_of_year: int,
                            elevation: np.ndarray | None =None,
                            d0: np.ndarray | None = None) -> np.ndarray:
      
      longitude_west = -longitude

      if d0 is None:
            if elevation is None:
                  """Eqs. 1 & 2, FCFDG 1992"""
                  latn = 46 + 23.4 * np.exp(-0.036 * (150 - longitude_west))
                  d0 = 151 * latitude / latn
            else:
      
                  """Eqs. 3 & 4, FCFDG 1992"""
                  latn = 43 + 33.7 * np.exp(-0.0351 * (150 - longitude_west))
                  d0 = 142.1 * (latitude/latn) + 0.0172 * elevation
      
      d0 = np.round(d0, 0)
      # date = datetime.strptime(day_of_year, "%Y-%m-%d")
      # dj = date.timetuple().tm_yday
      dj = int(day_of_year)
      nd = np.abs(dj - d0)

      fmc = np.full_like(latitude, np.nan, dtype=float)
      """Eqs. 6, 7 & 8, FCFDG 1992"""
      cond1 = nd < 30
      cond2 = (30 <= nd) & (nd <= 50)
      cond3 = nd > 50
      fmc[cond1] = 85 + 0.0189 * nd[cond1] ** 2
      fmc[cond2] = 32.9 + 3.17 * nd[cond2] - 0.0288 * nd[cond2] ** 2
      fmc[cond3] = 120
      

      return fmc