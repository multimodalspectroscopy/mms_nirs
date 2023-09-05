from dataclasses import dataclass

import numpy as np

boundaries = np.array(
    [
        [1, 20, 20, 1, 3],
        [0.970000000000000, 0, 0, 0, 0],
        [1, 40, 40, 2, 4],
    ]
)


@dataclass
class Boundaries:
    boundaries = boundaries
    columns = ["water_frac", "HHb", "HbO2", "a", "b"]
