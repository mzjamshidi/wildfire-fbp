import pytest
import pandas as pd
import numpy as np

from fbp.core.ros import initial_spread_index

ref_isi_data = pd.read_csv("tests/data/InitialSpreadIndex.csv").to_dict(orient="records")

@pytest.mark.parametrize("row",ref_isi_data)
def test_initial_spread_index(row):
    ffmc = row["ffmc"]
    ws = row["ws"]
    fbpMod = row["fbpMod"]
    if not fbpMod:
        pytest.skip("Skipping test case where fbpMod is False.")
    
    
    isi = initial_spread_index(ffmc, ws)
    isi_ref = row["InitialSpreadIndex"]
    
    assert np.isclose(isi, isi_ref, atol=1e-2), f"ISI mismatch for input {row}"
    




