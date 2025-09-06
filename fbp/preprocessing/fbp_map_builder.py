from typing import Self

import numpy as np
from skimage import measure, morphology

from fbp.constants import FBP_FUEL_MAP

DECIDUOUS_FUEL_CODES = [code for fuel, code in FBP_FUEL_MAP.items() if fuel.startswith("D")]
CONIFER_FUEL_CODES = [code for fuel, code in FBP_FUEL_MAP.items() if fuel.startswith("C")]

class FuelMapBuilder:
    def __init__(self, vegetation_map: np.ndarray, kernel=10, background_index=0) -> None:
        self.kernel = kernel
        self.background_index = background_index
        
        self.vegetation_map = vegetation_map.copy()

        self.fuel_mask = (self.vegetation_map != background_index)
        
        """NRCan FBP Fuel Layer 2018 (pp. 1)"""
        self.fuel_layer = self._reduce_mean(self.fuel_mask) >= 0.6

        self.fbp_fuel_layer = np.zeros_like(self.fuel_layer, dtype=int)

   
    def map_vegetation_to_fuel(self, mapping: dict) -> Self:
        self.fuel_map = np.zeros_like(self.vegetation_map, dtype=int)
        for veg_code, fuel_code in mapping.items():
            self.fuel_map[self.vegetation_map == veg_code] = FBP_FUEL_MAP[fuel_code]    
        return self

    def compute_crown_closure(self, threshold=0.10) -> Self:
        """NRCan FBP Fuel Layer 2018 (pp. 1): forest vs. open thershold = 0.10"""
        cc = np.isin(self.fuel_map, DECIDUOUS_FUEL_CODES + CONIFER_FUEL_CODES)
        self.crown_closure_mask = cc
        self.forest_layer = self._reduce_mean(cc) >= threshold
        return self
    
    def compute_deciduous(self, threshold=0.75) -> Self:
        """NRCan FBP Fuel Layer 2018 (pp. 1): deciduous vs. conifer thershold = 0.75"""
        dec_mask = np.isin(self.fuel_map, DECIDUOUS_FUEL_CODES)
        deciduous_ratio_layer = self._reduce_sum(dec_mask) / (self._reduce_sum(self.crown_closure_mask) + 1e-8)
        deciduous_layer = deciduous_ratio_layer >= threshold
        self.fbp_fuel_layer[deciduous_layer] = self._reduce_majority(self.fuel_map[dec_mask])
        self.deciduous_ratio_layer = deciduous_ratio_layer
        self.deciduous_layer = deciduous_layer
        return self
    
    def compute_conifer(self, threshold=0.75) -> Self:
        """NRCan FBP Fuel Layer 2018 (pp. 1): deciduous vs. conifer thershold = 0.75"""
        conf_mask = np.isin(self.fuel_map, CONIFER_FUEL_CODES)
        conifer_ratio_layer = self._reduce_sum(conf_mask) / (self._reduce_sum(self.crown_closure_mask) + 1e-8)
        conifer_layer = conifer_ratio_layer >= threshold
        self.fbp_fuel_layer[conifer_layer] = self._reduce_majority(self.fuel_map[conf_mask])
        self.conifer_ratio_layer = conifer_ratio_layer
        self.conifer_layer = conifer_layer
        return self
    
    def compute_mixedwood(self, kind="M1") -> Self:
        mixed_mask = (~self.conifer_layer) & (~self.deciduous_layer) & (self.forest_layer)
        self.fbp_fuel_layer[mixed_mask] = FBP_FUEL_MAP[kind]
        self.mixedwood_layer = mixed_mask
        return self
    
    def compute_open_stands(self, kind="O1a") -> Self:
        open_mask = (self.fuel_layer) & (~self.forest_layer)
        self.fbp_fuel_layer[open_mask] = FBP_FUEL_MAP[kind]
        self.open_stand = open_mask
        return self

   
    def _reduce_mean(self, layer: np.ndarray) -> np.ndarray:
        return measure.block_reduce(
            layer, block_size=self.kernel, func=np.mean
        )
    
    def _reduce_sum(self, layer: np.ndarray) -> np.ndarray:
        return measure.block_reduce(
            layer, block_size=self.kernel, func=np.sum
        )

    def _reduce_majority(self, layer: np.ndarray) -> np.ndarray:
        def majority_func(arr, axis=None):
            if arr.size == 0:
                return 0
            values, counts = np.unique(arr, return_counts=True)
            mode = values[np.argmax(counts)]
            return mode

        return measure.block_reduce(
            layer, block_size=self.kernel, func=majority_func
        )
        
    # def close_standing_vegetation(self, radius=10) -> Self:
    #     self.standing_map = morphology.closing(self.standing_map, morphology.disk(radius))
    #     return self
    
    # def separte_stading_by_patch(self) -> Self:
    #     self.labeled_standing_map = measure.label(self.standing_map, connectivity=2)
    #     return self
    
       
    def get_fbp_fuel_layer(self) -> np.ndarray:
        return self.fbp_fuel_layer
    

