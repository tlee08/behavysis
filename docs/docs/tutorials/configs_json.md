# Configs JSON File

A configs JSON file is attached to each experiment.
This file defines a) how the experiment should be processed (e.g. hyperparameters like the `dlc_config_fp` to use), and b) the inherent parameters of the experiment (e.g. like the `px/mm` and `start_frame` calculations).

An example configs file is shown below:

```json
{
  "user": {
    "format_vid": {
      "height_px": 540,
      "width_px": 960,
      "fps": 15,
      "start_sec": null,
      "stop_sec": null
    },
    "run_dlc": {
      "model_fp": "/path/to/dlc_config.yaml"
    },
    "calculate_params": {
      "start_frame": {
        "window_sec": 1,
        "pcutoff": 0.9,
        "bodyparts": "--bodyparts-simba"
      },
      "exp_dur": {
        "window_sec": 1,
        "pcutoff": 0.9,
        "bodyparts": "--bodyparts-simba"
      },
      "stop_frame": {
        "dur_sec": 6000
      },
      "px_per_mm": {
        "pt_a": "--tl",
        "pt_b": "--tr",
        "dist_mm": 400
      }
    },
    "preprocess": {
      "interpolate": {
        "pcutoff": 0.5
      },
      "bodycentre": {
        "bodyparts": "--bodyparts-centre"
      },
      "refine_ids": {
        "marked": "mouse1marked",
        "unmarked": "mouse2unmarked",
        "marking": "AnimalColourMark",
        "window_sec": 0.5,
        "metric": "rolling",
        "bodyparts": "--bodyparts-centre"
      }
    },
    "evaluate": {
      "keypoints_plot": {
        "bodyparts": ["Nose", "BodyCentre", "TailBase1"]
      },
      "eval_vid": {
        "funcs": ["keypoints", "behavs"],
        "pcutoff": 0.5,
        "colour_level": "individuals",
        "radius": 4,
        "cmap": "rainbow"
      }
    },
    "extract_features": {
      "individuals": ["mouse1marked", "mouse2unmarked"],
      "bodyparts": "--bodyparts-simba"
    },
    "classify_behaviours": [
      {
        "model_fp": "/path/to/behav_model_1.json",
        "pcutoff": null,
        "min_window_frames": "--min_window_frames",
        "user_behavs": "--user_behavs"
      },
      {
        "model_fp": "/path/to/behav_model_2.json",
        "pcutoff": null,
        "min_window_frames": "--min_window_frames",
        "user_behavs": "--user_behavs"
      }
    ],
    "analyse": {
      "thigmotaxis": {
        "thresh_mm": 50,
        "roi_top_left": "--tl",
        "roi_top_right": "--tr",
        "roi_bottom_left": "--bl",
        "roi_bottom_right": "--br",
        "bodyparts": "--bodyparts-centre"
      },
      "center_crossing": {
        "thresh_mm": 125,
        "roi_top_left": "--tl",
        "roi_top_right": "--tr",
        "roi_bottom_left": "--bl",
        "roi_bottom_right": "--br",
        "bodyparts": "--bodyparts-centre"
      },
      "in_roi": {
        "thresh_mm": 5,
        "roi_top_left": "--tl",
        "roi_top_right": "--tr",
        "roi_bottom_left": "--bl",
        "roi_bottom_right": "--br",
        "bodyparts": ["Nose"]
      },
      "speed": {
        "smoothing_sec": 1,
        "bodyparts": "--bodyparts-centre"
      },
      "social_distance": {
        "smoothing_sec": 1,
        "bodyparts": "--bodyparts-centre"
      },
      "freezing": {
        "window_sec": 2,
        "thresh_mm": 5,
        "smoothing_sec": 0.2,
        "bodyparts": "--bodyparts-simba"
      },
      "bins_sec": [30, 60, 120, 300],
      "custom_bins_sec": [60, 120, 300, 600]
    }
  },
  "ref": {
    "bodyparts-centre": [
      "LeftFlankMid",
      "BodyCentre",
      "RightFlankMid",
      "LeftFlankRear",
      "RightFlankRear",
      "TailBase1"
    ],
    "bodyparts-simba": [
      "LeftEar",
      "RightEar",
      "Nose",
      "BodyCentre",
      "LeftFlankMid",
      "RightFlankMid",
      "TailBase1",
      "TailTip4"
    ],
    "tl": "TopLeft",
    "tr": "TopRight",
    "bl": "BottomLeft",
    "br": "BottomRight",
    "min_window_frames": 2,
    "user_behavs": ["fight", "aggression"]
  }
}
```

## The Structure

The configs file has three main sections
- `user`: User defined parameters to process the experiment.
- `auto`: Automatically calculated parameters which are used
    in later processes for the experiment.
    Also gives useful insights into how the experiment "went" (e.g. over/under time, arena is smaller than other videos).
- `ref`: User defined parameters can be referenced from keys defined here.
    Useful when the same parameter values are used for many processes
    (e.g. bodyparts).

## Understanding Specific Parameters

!!! Notes

    To understand specific parameters in the `configs.yaml`,
    see each processing function's API documentation.

    For example, `user.calculate_params.px_per_mm` requires
    `pt_a`, `pt_b`, and `dist_mm`, which are described in the
    [API docs][behavysis.processes.CalculateParams].

## The Ref section

The `ref` section defines values that can be referenced in the `user` section.

To reference a value from the ref section, first define it:

```json
{
    ...
    "ref": {
        "example": ["values", "of", "any", "type"]
    }
}
```

You can now reference `example` by prepending a double hyphen (`--`) when referencing it:

```json
{
    "user": {
        ...
        "parameter": "--example",
        ...
    },
    ...
}
```

## Setting the Configs for an Experiment or all Experiments in a Project

Each experiment requires a corresponding configs file.

To generate or modify an experiment's configs file, first make a `default.json` file with the configs configured as you'd like.

!!! Tip

    You can copy the example configs file from [here][configs-json-file].

    Just make sure to change the multiple `model_fp` filepaths.

```py
from behavysis import Experiment

# Getting the experiment
experiment = Experiment("exp_name", "root_dir")
# Making/overwriting the configs file
experiment.update_configs("/path/to/default.json", overwrite="all")
```

!!! Note

    The `overwrite` keyword can be `"all"`, or `"user"`.