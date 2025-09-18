import pytest
import pandas as pd
import numpy as np

from fbp.constants import FBP_FUEL_MAP
from fbp.core.ros import initial_spread_index, rate_of_spread, initial_rate_of_spread, buildup_effect
from fbp.core.slope import slope_adjusted_wind_vector
from fbp.core.weather import foliar_moisture_content

ref_isi_data = pd.read_csv("tests/data/InitialSpreadIndex.csv").to_dict(orient="records")
ref_slope_data = pd.read_csv("tests/data/Slope.csv").to_dict(orient="records")
ref_rate_of_spread = pd.read_csv("tests/data/RateOfSpread.csv").to_dict(orient="records")
ref_folier_moisture_content = pd.read_csv("tests/data/FoliarMoistureContent.csv").to_dict(orient="records")

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

@pytest.mark.parametrize("row", ref_folier_moisture_content)
def test_foliar_moisture_content(row):
    lat = np.asarray(row["LAT"], dtype=float)
    lon = np.asarray(row["LONG"], dtype=float)
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
