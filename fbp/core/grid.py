import numpy as np

class Grid:
    def __init__(self, fuel_map, slope_map=None, moister_map=None) -> None:
        # TODO add docstring
        
        self.fuel_map = fuel_map.astype(int)
        self.slope_map = slope_map if slope_map is not None else np.zeros_like(fuel_map)
        self.moisture_map = moister_map if moister_map is not None else np.zeros_like(fuel_map)

    
    def __repr__(self) -> str:
        return f"Grid(shape={self.fuel_map.shape})"
    

if __name__ == "__main__":
    fuel_map = np.array([
        [1, 1, 2, 2, 3],
        [1, 2, 2, 3, 3],
        [1, 1, 1, 2, 2],
        [2, 2, 1, 1, 1],
        [3, 2, 1, 1, 1]
    ])

    grid = Grid(fuel_map)
    breakpoint()