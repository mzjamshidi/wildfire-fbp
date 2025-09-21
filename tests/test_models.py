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
    dc_yesterday = 15.

    fwi_model = FWIModel(latitude_north=lat,
                         latitude_south=lat,
                         longitude_east=long,
                         longitude_west=long,
                         shape=(1, 1))
    
    for row in ref_fwi:
        yr, mon, day = row["YR"], row["MON"], row["DAY"]
        temp, prec, rh = row["TEMP"], row["PREC"], row["RH"]

        results = fwi_model.run(date=datetime(yr, mon, day),
                    temperature=temp,
                    precipitation=prec,
                    relative_humidity=rh,
                    duff_moisture_code_yesterday=dmc_yesterday,
                    drought_code_yesterday=dc_yesterday,
                    )
    
        ref_bui = row["BUI"]
        ref_dmc = row["DMC"]
        ref_dc = row["DC"]

        assert np.isclose(
            results.bui_today, ref_bui, atol=1e-2
        ), f"BUI mismatch: got {results.bui_today}, expected {ref_bui}"

        assert np.isclose(
            results.dmc_today, ref_dmc, atol=1e-2
        ), f"DMC mismatch: got {results.dmc_today}, expected {ref_dmc}"

        assert np.isclose(
            results.dc_today, ref_dc, atol=1e-2
        ), f"DC mismatch: got {results.dc_today}, expected {ref_dc}"

        dmc_yesterday = results.dmc_today
        dc_yesterday = results.dc_today
        
    

