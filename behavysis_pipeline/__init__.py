"""
This package is used to perform Behavioural Analytics (BA) on lab mice using computer vision.
The package allows users to perform the entire analytics pipeline from raw lab footage to
interpretable plotted and tabulated data for different analysises. This pipeline includes:

- Formatting raw videos to a desired mp4 format (e.g. user defined fps and resolution)
- Performing stance detection on the mp4 file to generate an annotated mp4 file and file that tabulates the x-y coordinates of the subject's body points in each video frame. DeepLabCut is used to perform this.
- Preprocessing the coordinates file
- Extracting meaningful data analysis from the preprocessed coordinates file
"""

#####################################################################
#               FILTERING STDOUT WARNINGS
#####################################################################

import warnings

warnings.filterwarnings("ignore")

#####################################################################
#         IMPORTING MODULES (INCL. RELATIVE AND 3RD PARTY)
#####################################################################

from behavysis_pipeline.pipeline import BAExperiment, BAProject
