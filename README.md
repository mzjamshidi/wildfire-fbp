# wildfire-fbp

`wildfire-fbp` is a **Python implementation of the Fire Behavior Prediction (FBP) system**, inspired by the R package [cffdrs](https://github.com/cffdrs/cffdrs_r), developed by **Natural Resources Canada (NRCan)**. It provides tools to calculate **fire danger indices**, making it suitable for wildfire risk assessment and research.

For interactive examples and step-by-step demos, see the [`notebooks/`](notebooks) folder.

---

## Key Features

- Vectorized implementation with **NumPy** for fast calculations.
- Python-native interface for integration with GIS, remote sensing, or AI-driven modeling workflows.
- Consistent with the original **Canadian Forest Fire Weather Index System**.


---

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/mzjamshidi/wildfire-fbp.git
cd wildfire-fbp
```

---


## References

- Forestry Canada Fire Danger Group. (1992). *Development and Structure of the Canadian Forest Fire Behavior Prediction System*. Ottawa: Forestry Canada.  
- Van Wagner, C. E., & Pickett, T. L. (1985). *Equations and FORTRAN Program for the Canadian Forest Fire Weather Index System*. Chalk River, Ontario: Canadian Forestry Service.  
- Wang, X., Wotton, B. M., Cantin, A. S., Parisien, M.-A., Anderson, K., Moore, B., & Flannigan, M. D. (2017). *cffdrs: An R package for the Canadian Forest Fire Danger Rating System*. Ecological Processes. [10.1186/s13717-017-0070-z](https://doi.org/10.1186/s13717-017-0070-z)
