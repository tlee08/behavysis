"""
_summary_
"""

FOLDERS = {
    "0_configs": ".json",
    "1_raw_vid": ".mp4",
    "2_formatted_vid": ".mp4",
    "3_dlc": ".feather",
    "4_preprocessed": ".feather",
    "5_features_extracted": ".feather",
    "6_predicted_behavs": ".feather",
    "7_scored_behavs": ".feather",
}

DIAGNOSTICS_DIR = "diagnostics"
ANALYSIS_DIR = "analysis"
EVALUATE_DIR = "evaluate"
TEMP_DIR = ".temp"


DLC_COLUMN_NAMES = ("scorer", "individuals", "bodyparts", "coords")
ANALYSIS_COLUMN_NAMES = ("individuals", "measures")
ANALYSIS_INDEX_NAMES = ("frame", "timestamp")

BEHAV_PROB_COL = "prob"
BEHAV_PRED_COL = "pred"
BEHAV_ACTUAL_COL = "actual"
BEHAV_COLUMN_NAMES = ("behaviours", "outcomes")

HDF_KEY = "data"

BODYCENTRE = "Centre"

# NEED TO CHANGE THIS FOR DIFFERENT NAMES OF CORNERS
ARENAPARTS = ("TopRight", "TopLeft", "BottomRight", "BottomLeft")

SINGLE_COL = "single"
PROCESS_COL = "processed"

GPUNUM = 0

PROCS = 2

DIAGNOSTICS_SUCCESS_MESSAGES = (
    "Success! Success! Success!!",
    "Done and DONE!!",
    "Yay! Completed!",
    "This process was completed. Good on you :)",
    "Thumbs up!",
    "Woohoo!!!",
    "Phenomenal!",
    ":) :) :) :) :)",
    "Go you!",
    "You are doing awesome!",
    "You got this!",
    "You're doing great!",
    "Sending good vibes.",
    "I believe in you!",
    "You're a champion!",
    "No task too tall :) :)",
    "A job done well, and a well done job!",
)
STR_DIV = "".ljust(50, "-")

PLOT_STYLE = "whitegrid"
PLOT_DPI = 75
