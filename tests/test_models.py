import pytest
from datetime import datetime

import numpy as np
import pandas as pd

from fbp import FWIModel


def test_fwi_van_wagner_calibration():

    ref_fwi = pd.read_csv("tests/data/fwi_01.csv").to_dict(orient="records")
    long = ref_fwi[0]["LONG"]
    lat = ref_fwi[0]["LAT"]

    dmc_yesterday = 6.0
    dc_yesterday = 15.0
    ffmc_yesterday = 85.0


    fwi_model = FWIModel(latitude_north=lat,
                         latitude_south=lat,
                         longitude_east=long,
                         longitude_west=long,
                         shape=(1, 1))
    
    for row in ref_fwi:
        yr, mon, day = row["YR"], row["MON"], row["DAY"]
        temp, prec, rh, ws = row["TEMP"], row["PREC"], row["RH"], row["WS"]

        results = fwi_model.run(date=datetime(yr, mon, day),
                    wind_speed=ws,
                    temperature=temp,
                    precipitation=prec,
                    relative_humidity=rh,
                    duff_moisture_code_yesterday=dmc_yesterday,
                    drought_code_yesterday=dc_yesterday,
                    fine_fuel_moisture_code_yesterday=ffmc_yesterday
                    )
    
        ref_bui = row["BUI"]
        ref_dmc = row["DMC"]
        ref_dc = row["DC"]
        ref_ffmc = row["FFMC"]
        ref_isi = row["ISI"]
        ref_fwi = row["FWI"]

        assert np.isclose(
            results.bui_today, ref_bui, atol=1e-2
        ), f"BUI mismatch: got {results.bui_today}, expected {ref_bui}"

        assert np.isclose(
            results.dmc_today, ref_dmc, atol=1e-2
        ), f"DMC mismatch: got {results.dmc_today}, expected {ref_dmc}"

        assert np.isclose(
            results.dc_today, ref_dc, atol=1e-2
        ), f"DC mismatch: got {results.dc_today}, expected {ref_dc}"

        assert np.isclose(
            results.ffmc_today, ref_ffmc, atol=1e-2 
        ), f"FFMC mismatch: got {results.ffmc_today}, expected {ref_ffmc}"

        assert np.isclose(
            results.ffmc_today, ref_ffmc, atol=1e-2 
        ), f"FFMC mismatch: got {results.ffmc_today}, expected {ref_ffmc}"

        assert np.isclose(
            results.isi_today, ref_isi, atol=1e-2 
        ), f"ISI mismatch: got {results.isi_today}, expected {ref_isi}"

        assert np.isclose(
            results.fwi_today, ref_fwi, atol=1e-2 
        ), f"ISI mismatch: got {results.fwi_today}, expected {ref_fwi}"

        dmc_yesterday = results.dmc_today
        dc_yesterday = results.dc_today
        ffmc_yesterday = results.ffmc_today
    

