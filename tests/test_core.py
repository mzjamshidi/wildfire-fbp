import pytest
import pandas as pd
import numpy as np

from fbp.constants import FBP_FUEL_MAP
from fbp.core.ros import rate_of_spread, initial_rate_of_spread, buildup_effect
from fbp.core.slope import slope_adjusted_wind_vector
from fbp.core.weather import foliar_moisture_content, duff_moisture_code, drought_code, builtup_index, fire_weather_index, initial_spread_index, fine_fuel_moisture_code

ref_isi_data = pd.read_csv("tests/data/InitialSpreadIndex.csv").to_dict(orient="records")
ref_slope_data = pd.read_csv("tests/data/Slope.csv").to_dict(orient="records")
ref_rate_of_spread = pd.read_csv("tests/data/RateOfSpread.csv").to_dict(orient="records")
ref_folier_moisture_content = pd.read_csv("tests/data/FoliarMoistureContent.csv").to_dict(orient="records")
ref_duff_moisture_code = pd.read_csv("tests/data/DuffMoistureCode.csv").to_dict(orient="records")
ref_drought_code = pd.read_csv("tests/data/DroughtCode.csv").to_dict(orient="records")
ref_buildup_index = pd.read_csv("tests/data/BuildupIndex.csv").to_dict(orient="records")
ref_fire_weather_index = pd.read_csv("tests/data/FireWeatherIndex.csv").to_dict(orient="records")
ref_fine_fuel_moisture_code = pd.read_csv("tests/data/FineFuelMoistureCode.csv").to_dict(orient="records")


def _to_arr(val, dtype=float):
    return np.asarray(val, dtype=dtype)

@pytest.mark.parametrize("row", ref_isi_data)
def test_initial_spread_index(row):
    ffmc = row["ffmc"]
    ws = row["ws"]
    fbpMod = row["fbpMod"]
    if not fbpMod:
        pytest.skip("Skipping test case where fbpMod is False.")
    
    
    isi = initial_spread_index(ffmc, ws)
    isi_ref = row["InitialSpreadIndex"]
    
    assert np.isclose(isi, isi_ref, atol=1e-2), f"ISI mismatch for input {row}"

@pytest.mark.parametrize("row", ref_rate_of_spread)
def test_rate_of_spread(row):
    fuel = row["FUELTYPE"]
    fuel = fuel[0].upper() + fuel[1:].lower()
    fuel_map=np.array(FBP_FUEL_MAP.get(fuel, 0))
    isi = np.array(row["ISI"], dtype=float)
    bui = np.array(row["BUI"], dtype=float)
    pc = np.array(row["PC"], dtype=float)
    pdf = np.array(row["PDF"], dtype=float)
    cc = np.array(row["CC"], dtype=float)

    if fuel.upper() in ["NF", "WA", "C6"]:
        pytest.skip(f"Skipping test case for {fuel}.")

    rsi = initial_rate_of_spread(fuel_map=fuel_map,
                                 isi=isi,
                                 percent_conifer_map=pc,
                                 percent_dead_fir_map=pdf,
                                 percent_grass_curing_map=cc)
    be = buildup_effect(fuel_map=fuel_map, bui=bui)
    ros = rate_of_spread(rsi=rsi, be=be)

    ref_ros = row["RateOfSpread"]

    assert np.allclose(ros, ref_ros, atol=1e-2, equal_nan=True), (
        f"Rate of spread mismatch for fuel '{fuel}': "
        f"computed={ros}, reference={ref_ros}"
    )

@pytest.mark.parametrize("row", ref_fine_fuel_moisture_code)
def test_fine_fuel_moisture_code(row):
    ffmc_yda = _to_arr(row["ffmc_yda"])
    temp = _to_arr(row["temp"])
    rh = _to_arr(row["rh"])
    ws = _to_arr(row["ws"])
    prec = _to_arr(row["prec"])

    ffmc = fine_fuel_moisture_code(ffmc_yesterday=ffmc_yda,
                                   temp=temp,
                                   rh=rh,
                                   prec=prec,
                                   ws=ws)
    
    ref_ffmc = row["FineFuelMoistureCode"]

    assert np.allclose(
            ffmc, ref_ffmc, atol=1e-2, equal_nan=True
        ), f"BUI mismatch: got {ffmc}, expected {ref_ffmc}"

@pytest.mark.parametrize("row", ref_folier_moisture_content)
def test_foliar_moisture_content(row):
    lat = np.asarray(row["LAT"], dtype=float)
    lon = np.asarray(row["LONG"], dtype=float) * -1
    elv = np.asarray(row["ELV"], dtype=float)
    dj = row["DJ"]
    d0_val = row["D0"]
    d0 = None if d0_val == 0 else np.asarray(d0_val, dtype=float)

    elv = None if elv == 0 else elv

    fmc = foliar_moisture_content(latitude=lat,
                                  longitude=lon,
                                  elevation=elv,
                                  day_of_year=dj,
                                  d0=d0)
    
    ref_fmc = row["FoliarMoistureContent"]

    assert np.isclose(fmc, ref_fmc, atol=1), f"FMC mismatch for row {row}"

@pytest.mark.parametrize("row", ref_duff_moisture_code)
def test_duff_code(row):
    dmc_yda = _to_arr(row["dmc_yda"])
    temp = row["temp"]
    rh = _to_arr(row["rh"])
    prec = _to_arr(row["prec"])
    lat = _to_arr(row["lat"])
    mon = row["mon"]
    lat_adj = row["lat.adjust"]
    lat = lat if lat_adj else None

    dmc = duff_moisture_code(dmc_yesterday=dmc_yda,
                             temp=temp,
                             rh=rh,
                             prec=prec,
                             latitude=lat,
                             month=mon)
    
    ref_dmc = row["DuffMoistureCode"]
    assert np.isclose(dmc, ref_dmc, atol=1e-2), f"DMC mismatch for row {row}"

@pytest.mark.parametrize("row", ref_drought_code)
def test_drought_code(row):
    dc_yds = _to_arr(row["dc_yda"])
    temp = _to_arr(row["temp"])
    prec = _to_arr(row["prec"])
    lat = _to_arr(row["lat"])
    lat_adj = row["lat.adjust"]
    lat = lat if lat_adj else None
    mon = row["mon"]

    dc = drought_code(dc_yesterday=dc_yds,
                      temp=temp,
                      prec=prec,
                      month=mon,
                      latitude=lat)
    
    ref_dc = row["DroughtCode"]

    assert np.isclose(dc, ref_dc, atol=1e-2), f"DC mismatch for row {row}"

@pytest.mark.parametrize("row", ref_buildup_index)
def test_buildup_index(row):
    dmc = _to_arr(row["dmc"])
    dc = _to_arr(row["dc"])

    bui = builtup_index(dmc=dmc, dc=dc)

    ref_bui = row["BuildupIndex"]

    assert np.isclose(bui, ref_bui, atol=1e-2), f"BUI mismatch for row {row}"

@pytest.mark.parametrize("row", ref_fire_weather_index)
def test_fire_weather_index(row):
    isi = _to_arr(row["isi"])
    bui = _to_arr(row["bui"])

    fwi = fire_weather_index(isi=isi, bui=bui)

    ref_fwi = row["FireWeatherIndex"]

    assert np.isclose(fwi, ref_fwi, atol=1e-2), f"FQI mismatch for row {row}"



# @pytest.mark.parametrize("row", ref_slope_data)
# def test_slope(row):
#     fuel = row["FUELTYPE"]
#     fuel = fuel[0].upper() + fuel[1:].lower()
#     bui = row["BUI"]
#     ws = np.array(row["WS"], dtype=float)
#     waz = np.array(np.degrees(row["WAZ"]), dtype=float) 
#     gs = np.array(row["GS"], dtype=float)
#     saz = np.array(np.degrees(row["SAZ"]), dtype=float)
#     ffmc = np.maximum(row["FFMC"], 0)
#     pc = np.array(row["PC"], dtype=float)
#     pdf = np.array(row["PDF"], dtype=float)
#     cc = np.array(row["CC"], dtype=float)

#     if fuel not in ["D1"] or gs == 0 or np.isclose(saz, 257.5) and not np.isclose(waz, 257.5):
#         pytest.skip(f"Skipping test case for {fuel}.")


#     wsv, raz = slope_adjusted_wind_vector(fuel_map=np.array(FBP_FUEL_MAP.get(fuel, 0)),
#                                wind_speed=ws,
#                                wind_azimuth=waz,
#                                slope_percent=gs,
#                                slope_azimuth=saz,
#                                ffmc=ffmc,
#                                percent_conifer_map=pc,
#                                percent_grass_curing_map=cc,
#                                percent_dead_fir_map=pdf,
#                                )

#     ref_wsv = row["WSV"]
#     if ref_wsv == "NA":
#         ref_wsv = np.nan
#     ref_raz = np.degrees(row["RAZ"])

#     # print(ref_wsv, wsv)

#     # if "O" in fuel:
#     # breakpoint()
#     if not np.allclose(wsv, ref_wsv, atol=1e-1, equal_nan=True):
#         breakpoint()
    
#     assert np.allclose(wsv, ref_wsv, atol=1e-1, equal_nan=True)
#     # assert np.isclose(raz, ref_raz, atol=1e-1)