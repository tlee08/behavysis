# Project Setup Tutorial

This tutorial walks you through setting up a project folder for behavysis analysis. By the end, you'll have a properly structured folder ready for processing.

!!! tip "Prerequisites"
    - behavysis installed ([Installation Guide](../installation/installing.md))
    - One or more video files of laboratory mice

---

## Step 1: Create Your Project Folder

Create a new folder for your analysis. This folder will contain all your experiments.

```bash
mkdir my_behavior_project
cd my_behavior_project
```

Or use your file manager (Finder on Mac, File Explorer on Windows).

---

## Step 2: Understand the Folder Structure

behavysis uses a specific folder structure. Here's what you'll create:

![Project folder showing main structure](figures/folders1.png)

Each folder represents a stage in the analysis pipeline:

| Folder Number | Folder Name | Content |
|:-------------:|:------------|:--------|
| 0 | `0_configs/` | JSON configuration files |
| 1 | `1_raw_vid/` | Your original video files |
| 2 | `2_formatted_vid/` | Resized and standardized videos |
| 3 | `3_keypoints/` | DeepLabCut pose estimation output |
| 4 | `4_preprocessed/` | Cleaned and interpolated data |
| 5 | `5_features_extracted/` | Derived features for ML |
| 6 | `6_predicted_behavs/` | Automated behavior predictions |
| 7 | `7_scored_behavs/` | Human-verified behaviors |
| 8 | `8_analysis/` | Statistical results |
| 9 | `9_analysis_combined/` | Combined results across experiments |
| 10 | `10_evaluate_vid/` | Annotated evaluation videos |

!!! important "Auto-Created Folders"
    You only need to create folders `0_configs` and `1_raw_vid` manually. The rest are created automatically during processing.

---

## Step 3: Create Required Folders

```bash
mkdir 0_configs
mkdir 1_raw_vid
```

Your project folder should now look like:

```
my_behavior_project/
├── 0_configs/
└── 1_raw_vid/
```

---

## Step 4: Add Your Videos

Copy your video files into `1_raw_vid/`:

```bash
cp /path/to/your/videos/*.mp4 1_raw_vid/
```

### Naming Convention

Each experiment must have a unique name. behavysis uses the filename (without extension) to identify experiments.

???+ example "Good Names"
    ```
    mouse_A_day1.mp4
    mouse_A_day2.mp4
    mouse_B_day1.mp4
    batch_001_session1.mp4
    ```

???+ warning "Avoid These"
    ```
    video 1.mp4        # Has spaces
    my.video.name.mp4  # Multiple dots
    .hidden.mp4        # Starts with dot
    ```

![Folder with multiple experiment videos](figures/folders3.png)

---

## Step 5: Video Requirements

### Supported Formats

- **.mp4** (recommended)
- .avi, .mov (may work but not tested)

### Recommended Specifications

| Property | Recommendation |
|----------|---------------|
| Resolution | 640×480 or higher |
| Frame Rate | 15-30 FPS |
| Lighting | Consistent, no flicker |
| Background | Uniform color (helps DLC) |
| Duration | 5-60 minutes typical |

### Multi-Animal Videos

behavysis supports multi-animal tracking. Your videos should have:

- Clear view of all animals
- Distinguishable markings (if applicable)
- Minimal occlusion (overlapping)

---

## Step 6: Create a Default Configuration

The configuration file tells behavysis how to process your videos. Create a file called `default_config.json`:

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
      "model_fp": "/path/to/your/dlc_config.yaml"
    },
    "analyse": {
      "thigmotaxis": {
        "thresh_mm": 50,
        "roi_top_left": "--tl",
        "roi_top_right": "--tr",
        "roi_bottom_left": "--bl",
        "roi_bottom_right": "--br",
        "bodyparts": "--bodyparts-centre"
      },
      "speed": {
        "smoothing_sec": 1,
        "bodyparts": "--bodyparts-centre"
      },
      "bins_sec": [30, 60, 120, 300]
    }
  },
  "ref": {
    "bodyparts-centre": [
      "BodyCentre",
      "LeftFlankMid",
      "RightFlankMid"
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
    "br": "BottomRight"
  }
}
```

!!! critical "Update Required"
    You **must** change `/path/to/your/dlc_config.yaml` to point to your actual DeepLabCut model configuration file.

---

## Step 7: Verify Your Setup

Let's verify everything is set up correctly with Python:

```python
from behavysis import Project

# Initialize project
proj = Project("./my_behavior_project")

# Import experiments
proj.import_experiments()

# Print discovered experiments
print(f"Found {len(proj.experiments)} experiments:")
for exp in proj.experiments:
    print(f"  - {exp.name}")
```

Expected output:
```
Found 3 experiments:
  - mouse_A_day1
  - mouse_A_day2
  - mouse_B_day1
```

If you see your experiments listed, your project is ready!

---

## Step 8: Apply Default Configuration

Apply the configuration to all experiments:

```python
from behavysis import Project

proj = Project("./my_behavior_project")
proj.import_experiments()

# Apply default config to all experiments
proj.update_configs(
    default_configs_fp="./default_config.json",
    overwrite="user"  # Only update user-defined params, preserve auto params
)
```

!!! note "Overwrite Options"
    - `"user"` — Update only user-defined parameters (recommended)
    - `"all"` — Update both user and auto parameters
    - `"reset"` — Replace entire config file with default

---

## Complete Example

Here's a complete setup script you can adapt:

```python
#!/usr/bin/env python
"""Setup script for behavysis project."""

import os
from behavysis import Project

# Configuration
PROJECT_DIR = "./my_behavior_project"
DEFAULT_CONFIG = "./default_config.json"

# Initialize and setup
proj = Project(PROJECT_DIR)
proj.import_experiments()

print(f"Project: {PROJECT_DIR}")
print(f"Experiments: {len(proj.experiments)}")

# Update configs if default exists
if os.path.exists(DEFAULT_CONFIG):
    proj.update_configs(DEFAULT_CONFIG, overwrite="user")
    print("✓ Configurations updated")
else:
    print("⚠ No default config found. Please create one.")

print("\nSetup complete! You're ready to:")
print("  1. Run: proj.format_vid(overwrite=True)")
print("  2. Run: proj.run_dlc(gputouse=0, overwrite=True)")
print("  3. Continue with preprocessing and analysis")
```

---

## Troubleshooting

### No experiments found

**Problem:** `import_experiments()` returns 0 experiments.

**Solutions:**
- Verify videos are in `1_raw_vid/` (not a subfolder)
- Check that video extensions are `.mp4`
- Ensure video filenames don't start with a dot (hidden files)

### "Folder does not exist" error

**Problem:** Error about missing folder when importing experiments.

**Solution:** You only need `0_configs` and `1_raw_vid`. Other folders are created automatically, but create them manually if you get errors:

```bash
mkdir -p 0_configs 1_raw_vid 2_formatted_vid 3_keypoints \
  4_preprocessed 5_features_extracted 6_predicted_behavs \
  7_scored_behavs 8_analysis 9_analysis_combined \
  10_evaluate_vid 0_diagnostics
```

### Config file errors

**Problem:** JSON errors when loading config.

**Solution:** Validate your JSON syntax:

```bash
python -c "import json; json.load(open('default_config.json'))"
```

---

## Next Steps

Your project is now ready! Continue with:

1. **[Configuration Tutorial](configs_json.md)** — Understand all config parameters
2. **[Analysis Example](../examples/analysis.md)** — Run your first analysis
3. **[Diagnostics Guide](diagnostics_messages.md)** — Understand processing outputs

---

## Summary Checklist

- [ ] Created project folder
- [ ] Created `0_configs/` folder
- [ ] Created `1_raw_vid/` folder
- [ ] Copied video files to `1_raw_vid/`
- [ ] Named files consistently (no spaces)
- [ ] Created `default_config.json`
- [ ] Updated DLC model path in config
- [ ] Ran `import_experiments()` successfully
- [ ] Applied default configuration
