# Config JSON File

A config JSON file is attached to each experiment. This file defines:

1. **How** the experiment should be processed (e.g., which DLC model to use)
2. **Inherent parameters** of the experiment (e.g., px/mm, start_frame)

[:material-book-open-variant: Full config reference](#full-config-reference){ .md-button }

---

## Quick Start

1. Generate a default config file:

   ```bash
   behavysis-make-project
   ```

2. Edit `default_config.json` - most experiments only need to change a few fields.

3. Apply config to all experiments:
   ```python
   proj.update_config(default_config_fp="default_config.json", overwrite="user")
   ```

---

## Required Settings

These fields **must** be configured for the pipeline to work:

| Field                                     | Description                                   | Example                          |
| ----------------------------------------- | --------------------------------------------- | -------------------------------- |
| `user.run_dlc.model_fp`                   | Path to DeepLabCut model config.yaml          | `"/models/my_model/config.yaml"` |
| `user.calculate_params.px_per_mm.dist_mm` | Real-world distance between two arena corners | `400` (for 40cm arena)           |

---

## Config Structure

The config has three sections:

### `user` - Your Settings

Parameters you define to control processing:

```json
{
  "user": {
    "format_vid": { "fps": 15, "width_px": 960, "height_px": 540 },
    "run_dlc": { "model_fp": "/path/to/model/config.yaml" },
    "preprocess": { "interpolate": { "pcutoff": 0.5 } }
  }
}
```

### `auto` - Auto-Calculated Values

Populated automatically during processing:

```json
{
  "auto": {
    "px_per_mm": 2.5,
    "start_frame": 150,
    "stop_frame": 9000,
    "formatted_vid": { "fps": 15.0, "width_px": 960, "height_px": 540 }
  }
}
```

### `ref` - Reusable References

Define values once, reference them with `--name`:

```json
{
  "ref": {
    "bpts_centre": ["BodyCentre", "TailBase1"],
    "bpts_simba": ["Nose", "LeftEar", "RightEar", "BodyCentre", "TailBase1"]
  },
  "user": {
    "analyse": {
      "speed": { "bodyparts": "--bpts_centre" },
      "freezing": { "bodyparts": "--bpts_centre" }
    }
  }
}
```

---

## Common Settings by Stage

### Video Formatting

```json
{
  "format_vid": {
    "width_px": 960,
    "height_px": 540,
    "fps": 15.0,
    "start_sec": null,
    "stop_sec": null
  }
}
```

### DeepLabCut

```json
{
  "run_dlc": {
    "model_fp": "/path/to/DEEPLABCUT_model/config.yaml"
  }
}
```

### Preprocessing

```json
{
  "preprocess": {
    "interpolate": { "pcutoff": 0.5 },
    "interpolate_stationary": [],
    "refine_ids": {
      "marked": "mouse1marked",
      "unmarked": "mouse2unmarked",
      "marking": "AnimalColourMark",
      "window_sec": 0.5,
      "bodyparts": "--bpts_centre"
    }
  }
}
```

### Feature Extraction

```json
{
  "extract_features": {
    "individuals": "--indivs_simba",
    "bodyparts": "--bpts_simba"
  }
}
```

### Behavioral Classification

```json
{
  "classify_behavs": [
    {
      "proj_dir": "path/to/project",
      "behav_name": "fighting",
      "pcutoff": 0.5,
      "min_empty_window_secs": 0.2
    }
  ]
}
```

### Analysis

```json
{
  "analyse": {
    "bins_sec": [30, 60, 120],
    "speed": { "smoothing_sec": 1, "bodyparts": "--bpts_centre" },
    "social_distance": { "smoothing_sec": 1, "bodyparts": "--bpts_centre" },
    "freezing": {
      "window_sec": 2,
      "thresh_mm": 5,
      "bodyparts": "--bpts_centre"
    },
    "in_roi": [
      {
        "roi_corners": "--bpts_corners",
        "bodyparts": "--bpts_front",
        "padding_mm": 0
      }
    ]
  }
}
```

---

## Example Config

### Open Field (Single Mouse)

Standard open field test with one mouse. Measures speed, freezing, and thigmotaxis.

```json
{
  "user": {
    "format_vid": {
      "width_px": 960,
      "height_px": 540,
      "fps": 15.0
    },
    "run_dlc": {
      "model_fp": "/path/to/open_field_dlc/config.yaml"
    },
    "calculate_params": {
      "from_likelihood": {
        "bodyparts": "--bpts_simba",
        "window_sec": 1.0,
        "pcutoff": 0.8
      },
      "stop_frame_from_dur": {
        "dur_sec": 600
      },
      "px_per_mm": {
        "pt_a": "TopLeft",
        "pt_b": "TopRight",
        "pcutoff": 0.5,
        "dist_mm": 400
      }
    },
    "preprocess": {
      "interpolate": { "pcutoff": 0.5 }
    },
    "extract_features": {
      "individuals": ["single"],
      "bodyparts": "--bpts_simba"
    },
    "analyse": {
      "bins_sec": [30, 60, 120],
      "speed": {
        "smoothing_sec": 1,
        "bodyparts": ["BodyCentre"]
      },
      "freezing": {
        "window_sec": 2,
        "thresh_mm": 5,
        "smoothing_sec": 0.2,
        "bodyparts": ["BodyCentre"]
      }
    }
  },
  "ref": {
    "bpts_simba": [
      "Nose",
      "LeftEar",
      "RightEar",
      "BodyCentre",
      "TailBase1",
      "TailTip"
    ]
  }
}
```

### Two-Mouse Social Interaction

Social interaction test with two mice (one marked, one unmarked). Includes social distance and individual tracking.

```json
{
  "user": {
    "format_vid": {
      "width_px": 960,
      "height_px": 540,
      "fps": 15.0
    },
    "run_dlc": {
      "model_fp": "/path/to/social_interaction_dlc/config.yaml"
    },
    "calculate_params": {
      "from_likelihood": {
        "bodyparts": "--bpts_simba",
        "window_sec": 1.0,
        "pcutoff": 0.8
      },
      "stop_frame_from_dur": {
        "dur_sec": 600
      },
      "px_per_mm": {
        "pt_a": "TopLeft",
        "pt_b": "TopRight",
        "pcutoff": 0.5,
        "dist_mm": 400
      }
    },
    "preprocess": {
      "interpolate": { "pcutoff": 0.5 },
      "refine_ids": {
        "marked": "mouse1marked",
        "unmarked": "mouse2unmarked",
        "marking": "AnimalColourMark",
        "window_sec": 0.5,
        "bodyparts": "--bpts_centre"
      }
    },
    "extract_features": {
      "individuals": "--indivs_simba",
      "bodyparts": "--bpts_simba"
    },
    "analyse": {
      "bins_sec": [30, 60, 120],
      "speed": {
        "smoothing_sec": 1,
        "bodyparts": "--bpts_centre"
      },
      "social_distance": {
        "smoothing_sec": 1,
        "bodyparts": "--bpts_centre"
      },
      "freezing": {
        "window_sec": 2,
        "thresh_mm": 5,
        "smoothing_sec": 0.2,
        "bodyparts": "--bpts_centre"
      }
    }
  },
  "ref": {
    "indivs_simba": ["mouse1marked", "mouse2unmarked"],
    "bpts_simba": [
      "Nose",
      "LeftEar",
      "RightEar",
      "BodyCentre",
      "LeftFlankMid",
      "RightFlankMid",
      "TailBase1",
      "TailTip"
    ],
    "bpts_centre": ["BodyCentre", "TailBase1"]
  }
}
```

---

## Troubleshooting

### "DLC model config not found"

Check that `user.run_dlc.model_fp` points to an existing `.yaml` file:

```json
{ "run_dlc": { "model_fp": "/absolute/path/to/config.yaml" } }
```

### "Width and height must be provided"

Run `proj.format_vid()` first - it populates `auto.formatted_vid` dimensions.

### "Bodyparts not found in keypoints data"

Your DLC model's bodypart names don't match the config. Check `ref.bpts_simba` matches your model's output.

---

## Full Config Reference

```json
{
  "user": {
    "format_vid": {
      "width_px": 960,
      "height_px": 540,
      "fps": 15.0,
      "start_sec": null,
      "stop_sec": null
    },
    "run_dlc": {
      "model_fp": "path/to/DEEPLABCUT_model/config.yaml"
    },
    "calculate_params": {
      "from_likelihood": {
        "bodyparts": "--bpts_simba",
        "window_sec": 1.0,
        "pcutoff": 0.8
      },
      "start_frame_from_csv": {
        "csv_fp": "path_to/start_times.csv",
        "name": null
      },
      "stop_frame_from_dur": {
        "dur_sec": 6000
      },
      "px_per_mm": {
        "pt_a": "pt_a",
        "pt_b": "pt_b",
        "pcutoff": 0.5,
        "dist_mm": 400
      }
    },
    "preprocess": {
      "interpolate": {
        "pcutoff": 0.5
      },
      "interpolate_stationary": [],
      "refine_ids": {
        "marked": "marked",
        "unmarked": "unmarked",
        "marking": "marking",
        "bodyparts": "--bpts_centre",
        "window_sec": 0.5,
        "metric": "current"
      }
    },
    "extract_features": {
      "individuals": "--indivs_simba",
      "bodyparts": "--bpts_simba"
    },
    "classify_behavs": [
      {
        "proj_dir": "path/to/project_dir",
        "behav_name": "behav_name",
        "pcutoff": -1,
        "min_empty_window_secs": 0.2,
        "user_defined": []
      }
    ],
    "analyse": {
      "bins_sec": [30, 60, 120],
      "custom_bins_sec": [60, 120, 300, 600],
      "speed": {
        "smoothing_sec": 1,
        "bodyparts": "--bpts_centre"
      },
      "social_distance": {
        "smoothing_sec": 1,
        "bodyparts": "--bpts_centre"
      },
      "freezing": {
        "window_sec": 2,
        "thresh_mm": 5,
        "smoothing_sec": 0.2,
        "bodyparts": "--bpts_centre"
      },
      "in_roi": [
        {
          "roi_name": "in_my_roi",
          "is_in": true,
          "padding_mm": 0,
          "roi_corners": "--bpts_corners",
          "bodyparts": "--bpts_front"
        }
      ]
    },
    "evaluate_vid": {
      "funcs": ["keypoints", "analysis"],
      "pcutoff": 0.8,
      "colour_level": "individuals",
      "radius": 3,
      "cmap": "rainbow",
      "padding": 30
    }
  },
  "auto": {
    "raw_vid": {
      "width_px": -1,
      "height_px": -1,
      "fps": -1.0,
      "total_frames": -1
    },
    "formatted_vid": {
      "width_px": -1,
      "height_px": -1,
      "fps": -1.0,
      "total_frames": -1
    },
    "px_per_mm": -1.0,
    "start_frame": -1,
    "stop_frame": -1,
    "dur_frames": -1
  },
  "ref": {}
}
```

For detailed parameter documentation, see each processing function's API docs.
