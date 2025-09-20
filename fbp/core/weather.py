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



"""Table 1, Van Wagner & Pickett 1985 & R package"""
DMC_Leff_VALUES = {
      "46N": [6.5, 7.5, 9, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8, 7, 6],
      "20N": [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
      "20S": [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
      "40S": [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10, 11.2, 11.8],
      "0": 12 * [9]
}

def _dmc_effective_day_length(month: int,
                              latitude: np.ndarray | None =None) -> np.ndarray:
     
     idx = month -1 

     if latitude is None:
          return np.array(DMC_Leff_VALUES["46N"][idx], dtype=float)
     
     Leff = np.full_like(latitude, np.nan, dtype=float)

     # These latitude adjustments are based on the R package
     # (corresponding equations not found in the main source)
     Leff[(latitude >= 30.) & (latitude <= 90.)] = DMC_Leff_VALUES["46N"][idx]
     Leff[(latitude >= 10.) & (latitude <= 30.)] = DMC_Leff_VALUES["20N"][idx]
     Leff[(latitude >= -10.) & (latitude <= 10.)] = DMC_Leff_VALUES["0"][idx]
     Leff[(latitude >= -30.) & (latitude <= -10.)] = DMC_Leff_VALUES["20S"][idx]
     Leff[(latitude >= -90.) & (latitude <= -30.)] = DMC_Leff_VALUES["40S"][idx]

     return Leff     

def duff_moisture_code(dmc_yesterday: np.ndarray,
                       temp: np.ndarray,
                       prec: np.ndarray,
                       rh: np.ndarray, 
                       month: int,
                       latitude: np.ndarray | None = None):
     
     pr = dmc_yesterday.copy()

     rainy = prec > 1.5
     if np.any(rainy):
          ra = prec[rainy]
          dmc_r = dmc_yesterday[rainy]
          """Eq. 11, Van Wagner & Pickett 1985"""
          rw = 0.92 * ra - 1.27
          """Eq. 12, Van Wagner & Pickett 1985 (per R package alterated to calculate more accurately)"""
          wmi = 20 + 280 / np.exp(0.023 * dmc_r)

          b = np.full_like(dmc_r, np.nan, dtype=float)
          
          """Eq. 13, Van Wagner & Pickett 1985"""
          cond1 = dmc_r <= 33
          cond2 = (dmc_r > 33) & (dmc_r <= 65)
          cond3 = dmc_r > 65
          
          b[cond1] = 100 / (0.5 + 0.3 * dmc_r[cond1])
          b[cond2] = 14 - 1.3 * np.log(dmc_r[cond2])
          b[cond3] = 6.2 * np.log(dmc_r[cond3]) - 17.2
          
          """Eq. 14, Van Wagner & Pickett 1985"""
          wmr = wmi + 1000 * rw / (48.77 + b * rw)
          
          """Eq. 15, Van Wagner & Pickett 1985 (per R package alterated to calculate more accurately)"""
          pr[rainy] = 43.43 * (5.6348 - np.log(wmr - 20))
      
     pr = np.maximum(pr, 0)
     
     """Van Wagner & Pickett 1985"""
     temp = np.maximum(temp, -1.1)
     
     Leff = _dmc_effective_day_length(month, latitude)
     
     """Eq. 16, Van Wagner & Pickett 1985"""
     rk = 1.894 * (temp + 1.1) * (100 - rh) * Leff * 1e-6

     """Eq. 17, Van Wagner & Pickett 1985"""
     dmc_today = pr + 100 * rk

     return dmc_today


"""Table 2, Van Wagner & Pickett 1985 & R package"""
DC_Leff_VALUES = {
      "20N": [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5, 2.4, 0.4, -1.6, -1.6],
      "20S": [6.4, 5, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8],
      "0": 12 * [1.4]
}

def _dc_effective_day_length(month: int,
                             latitude: np.ndarray | None = None):
      idx = month -1 

      if latitude is None:
            return np.array(DC_Leff_VALUES["20N"][idx], dtype=float)
     
      Leff = np.full_like(latitude, np.nan, dtype=float)

      # These latitude adjustments are based on the R package
      # (corresponding equations not found in the main source)
      Leff[(latitude >= 20)] = DC_Leff_VALUES["20N"][idx]
      Leff[(latitude <= -20)] = DC_Leff_VALUES["20S"][idx]
      Leff[(latitude < 20) & (latitude > -20)] = DC_Leff_VALUES["0"][idx]

      return Leff   


def drought_code(dc_yesterday: np.ndarray,
                 temp: np.ndarray,
                 prec: np.ndarray,
                 month: int,
                 latitude: np.ndarray | None = None):
     
      dr = dc_yesterday.copy()
      rainy = prec > 2.8
     
      if np.any(rainy):
            ra = prec[rainy]
            dc_ya = dc_yesterday[rainy]

            """Eq. 18, Van Wagner & Pickett 1985"""
            rw = 0.83 * ra - 1.27
            """Eq. 19, Van Wagner & Pickett 1985"""
            smi = 800 * np.exp(-dc_ya / 400)
            """Eq. 20, Van Wagner & Pickett 1985 (per R package alterated)"""
            dr0 = np.where(smi > 0,
                         dc_ya - 400 * np.log(1 + 3.937 * rw / smi),
                         0)
          
            dr[rainy] = np.maximum(dr0, 0)
      
      """Van Wagner & Pickett 1985"""
      temp = np.maximum(temp, -2.8)
      
      Leff = _dc_effective_day_length(month, latitude)
      
      """Eq. 22, Van Wagner & Pickett 1985"""
      pe = 0.36 * (temp + 2.8) + Leff
      pe = np.maximum(pe, 0)

      """Eq. 23, Van Wagner & Pickett 1985"""
      dc_today = dr + 0.5 * pe
      dc_today = np.maximum(dc_today, 0)

      return dc_today
      

def builtup_index(dmc: np.ndarray, dc: np.ndarray) -> np.ndarray:
     """
     dmc: duff moisture code
     dc: drought code
     """
     denom = (dmc + 0.4 * dc)
     safe_denom = np.where(denom == 0, 1e-6, denom)

     """Eq. 27, Van Wagner & Pickett 1985"""
     bui = np.where((dmc <= 0.4 * dc),
                    0.8 * dmc * dc / safe_denom,
                    dmc - (1 - 0.8 * dc / safe_denom) * (0.92 + (0.0114 * dmc) ** 1.7))
     
     bui[(dmc == 0) & (dc == 0)] = 0
     bui[bui < 0] = 0

     return bui

def fire_weather_index(isi: np.ndarray, bui: np.ndarray) -> np.ndarray:
     
     """Eq. 28, Van Wagner & Pickett 1985"""
     fD = np.where(bui <= 80,
                   0.626 * bui ** 0.809 + 2,
                   1000 / (25 + 108.64 * np.exp(-0.023* bui)))
     
     """Eq. 29, Van Wagner & Pickett 1985"""
     bb = 0.1 * isi * fD
     bb_safe = np.maximum(bb, 1)    # hack to avoid NaN warnings for bb < 1

     """Eq. 30, Van Wagner & Pickett 1985"""
     fwi = np.where(bb > 1,
                    np.exp(2.72 * (0.434 * np.log(bb_safe)) ** 0.647),
                    bb)
     return fwi

def _fF_formula(ffmc: np.ndarray):
    FFMC_COEFFICIENT = 250 * 59.5 / 101
    
    """Eq. 46, FCFDG 1992"""
    m = FFMC_COEFFICIENT * (101 - ffmc) / (59.5 + ffmc)

    """Eq. 45, FCFDG 1992"""
    return 91.9 * np.exp(-0.1386 * m) * (1 + (m**5.31) / 4.93e7)

def initial_spread_index(ffmc: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """
        ws: wind speed (km/h)
    """

    fF = _fF_formula(ffmc)
    
    """Eqs. 53 & 53a, FCFDG 1992: wsv: net effective wind speed"""
    fW = np.where(
        ws <= 40,
        np.exp(0.05039 * ws),
        12 * (1 - np.exp(-0.0818 * (ws - 28)))
    )

    """Eq. 52, FCFDG 1992"""
    isi = 0.208 * fW * fF
    return isi