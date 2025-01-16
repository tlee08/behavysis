import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MiscMixin:
    ###############################################################################################
    # Helper funcs (e.g. enum handling)
    ###############################################################################################

    @staticmethod
    def enum2tuple(my_enum):
        """
        Useful helper function to convert an Enum to a list of its values.
        Used in `check_df` and `init_df` functions.
        """
        return tuple(i.value for i in my_enum)

    @staticmethod
    def make_colours(vals, cmap: str) -> np.ndarray:
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
        # Returning
        return colours_ls
