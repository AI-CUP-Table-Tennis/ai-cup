from numpy import ndarray, dtype, float64, object_, int64
from typing import Literal

Double1D = ndarray[tuple[int], dtype[float64]]
DoubleNBy6 = ndarray[tuple[int, Literal[6]], dtype[float64]]
Object1D = ndarray[tuple[int], dtype[object_]]
Object2D = ndarray[tuple[int, int], dtype[object_]]
Axes1D = Object1D
Axes2D = Object2D
Long1D = ndarray[tuple[int], dtype[int64]]
