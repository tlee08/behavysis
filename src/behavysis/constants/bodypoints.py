"""Default bodypoint constants for SimBA feature extraction."""

INDIVS_SIMBA = [
    "mouse1marked",
    "mouse2unmarked",
]

BPTS_SIMBA = [
    "LeftEar",
    "RightEar",
    "Nose",
    "BodyCentre",
    "LeftFlankMid",
    "RightFlankMid",
    "TailBase1",
    "TailTip4",
]

# TODO: make agnostic from SimBA
BPMAP_SIMBA = {
    "LeftEar": "Ear_left",
    "RightEar": "Ear_right",
    "Nose": "Nose",
    "BodyCentre": "Center",
    "LeftFlankMid": "Lat_left",
    "RightFlankMid": "Lat_right",
    "TailBase1": "Tail_base",
    "TailTip4": "Tail_end",
}

BPTS_CENTRE = [
    "LeftFlankMid",
    "BodyCentre",
    "RightFlankMid",
    "LeftFlankRear",
    "RightFlankRear",
    "TailBase1",
]

BPTS_FRONT = ["LeftEar", "RightEar", "Nose", "BodyCentre"]

INDIVS_SINGLE = "single"

BPTS_CORNERS = [
    "TopLeft",
    "TopRight",
    "BottomRight",
    "BottomLeft",
]
