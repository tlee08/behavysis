from typing import Optional

import numpy as np

DIAGNOSTICS_SUCCESS_MESSAGES = (
    "Success! Success! Success!!\n",
    "Done and DONE!!\n",
    "Yay! Completed!\n",
    "This process was completed. Good on you :)\n",
    "Thumbs up!\n",
    "Woohoo!!!\n",
    "Phenomenal!\n",
    ":) :) :) :) :)\n",
    "Go you!\n",
    "You are doing awesome!\n",
    "You got this!\n",
    "You're doing great!\n",
    "Sending good vibes.\n",
    "I believe in you!\n",
    "You're a champion!\n",
    "No task too tall :) :)\n",
    "A job done well, and a well done job!\n",
    "Top job!\n",
)


def success_msg() -> str:
    """
    Return a random positive message :)
    """
    return np.random.choice(DIAGNOSTICS_SUCCESS_MESSAGES)


def file_exists_msg(fp: Optional[str] = None) -> str:
    """
    Return a warning message.
    """
    fp_str = f", {fp}, " if fp else " "
    return (
        f"WARNING: Output file"
        f"{fp_str}"
        "already exists - not overwriting file.\n"
        "To overwrite, specify overwrite=True`.\n"
    )
