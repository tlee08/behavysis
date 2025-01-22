import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavysis_pipeline.utils.logging_utils import init_logger

logger = init_logger(__name__)


def make_colours(vals, cmap: str) -> np.ndarray:
    """
    Given a vector of values and a colourmap,
    returns an array of colours (as RGBA arrays) corresponding
    to each unique value in the vals vector.

    Example
    -------
    ```
    [1,2,4,2,3,1,5]
    --> [Red, Blue, Green, Blue, Yellow, Red, Purple]
    ```
    """
    # If vals is an empty list, return colours_ls as an empty list
    if len(vals) == 0:
        return np.array([])
    # Encoding colours as 0, 1, 2, ... for each unique value
    colours_idx, _ = pd.factorize(vals)
    # Normalising to 0-1 (if only 1 unique value, it will be 0 div so setting values to 0)
    colours_idx = np.nan_to_num(colours_idx / colours_idx.max())
    # Getting corresponding colour for each item in `vals` list and from cmap
    colours_ls = plt.cm.get_cmap(cmap)(colours_idx)
    # Reassigning the order of the colours to be RGBA (not BGRA)
    colours_ls = colours_ls[:, [2, 1, 0, 3]]
    # Converting to (0, 255) range
    colours_ls = colours_ls * 255
    return colours_ls
