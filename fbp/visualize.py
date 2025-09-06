import matplotlib.pyplot as plt
import numpy as np

from .fbp import FBPResults

def plot_fire_intensity(results: FBPResults):
    fig, ax = plt.subplots()
    im = ax.imshow(results.hfi, cmap="Wistia")
    ax.set_title("Head Fire Intensity (kW/m)")
    cbar = fig.colorbar(im)
    plt.show()

def plot_rate_of_spread(results: FBPResults):
    fig, ax = plt.subplots()
    im = ax.imshow(results.ros, cmap="Wistia")
    ax.set_title("Rate of Spread (m/min)")
    cbar = fig.colorbar(im)
    plt.show()

def plot_fuel_consumption(results: FBPResults):
    fig, ax = plt.subplots()
    im = ax.imshow(results.tfc, cmap="Wistia")
    ax.set_title(r"Total Fuel Consumption (kg/m$^2$)")
    cbar = fig.colorbar(im)
    plt.show()

def plot_fuel_map(fuel_map: np.ndarray, extent=None):
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from .constants import FBP_FUEL_COLOR, FBP_FUEL_MAP, FBP_FUEL_DESC

    COLOR_MAP = {FBP_FUEL_MAP[f]: tuple(c/255 for c in color) for f, color in FBP_FUEL_COLOR.items()}

    FUEL_ID_TO_CODE = {c: f for f, c in FBP_FUEL_MAP.items()}
    
    fuel_map = fuel_map.copy()
    classes = np.unique(fuel_map)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    vectorized_map = np.vectorize(class_to_idx.get)(fuel_map)

    norm = BoundaryNorm(boundaries=np.arange(len(classes)+1)-0.5, ncolors=len(classes))

    colors = [COLOR_MAP[c] for c in classes]

    fig, ax = plt.subplots()
    im = ax.imshow(vectorized_map, cmap=ListedColormap(colors), norm=norm, extent=extent)
    cbar = fig.colorbar(im, ticks=np.arange(len(classes)))
    cbar.ax.set_yticklabels([f"{FUEL_ID_TO_CODE[f]}" for f in classes])

    plt.show()
