# Understanding the behavysis Workflow

This tutorial explains the conceptual framework behind behavysis — how it converts raw video footage into quantitative behavioral data. Understanding these concepts will help you use the tool more effectively.

!!! tip "Who is this for?"
    This tutorial is designed for researchers new to behavysis. No prior programming experience is required, though basic familiarity with Python will help.

---

## The Big Picture

behavysis automates what would otherwise be hours of manual video annotation. The typical workflow is:

1. **Record videos** of mice in experimental arenas
2. **Configure processing parameters** for your specific setup
3. **Run the pipeline** to extract pose data and analyze behaviors
4. **Review results** and validate with human annotation when needed

Let's explore each step.

---

## The Data Pipeline

behavysis processes your data through a series of stages, each building on the previous:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Raw Video  │────▶│ Format Video │────▶│  Pose Estimation │
│  (.mp4)     │     │ (resize, fps)│     │   (DeepLabCut)   │
└─────────────┘     └──────────────┘     └─────────────────┘
                                                       │
                                                       ▼
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Final Analysis  │◀────│ Behavior Analysis│◀────│ Feature         │
│  (statistics,    │     │ (detect actions) │     │ Extraction      │
│   plots)         │     │                  │     │ (velocities,    │
│                  │     │                  │     │  distances)     │
└──────────────────┘     └──────────────────┘     └─────────────────┘
                              ▲
                              │
┌──────────────────┐     ┌──────────────┐
│  Scored Videos   │────▶│  Train ML    │
│  (human verified)│     │  Classifiers │
└──────────────────┘     └──────────────┘
```

### Stage 1: Video Formatting

Raw videos from different cameras have different resolutions, frame rates, and codecs. behavysis standardizes them:

- **Resolution**: Scales to a consistent size (e.g., 960×540)
- **Frame rate**: Converts to a set FPS (e.g., 15 frames/second)
- **Codec**: Ensures compatibility with DeepLabCut

!!! note "Why standardize?"
    Pose estimation models are trained on specific video characteristics. Standardizing ensures consistent model performance across all your videos.

### Stage 2: Pose Estimation

DeepLabCut detects key anatomical landmarks in each frame:

| Body Part | What It Tracks |
|-----------|----------------|
| Nose | Front of the head, used for direction and sniffing |
| Ears | Head orientation |
| Body Centre | Overall body position |
| Flanks | Body shape and posture |
| Tail Base | Rear body position |
| Tail Tip | Tail movement |
| Corner Points | Arena boundaries (if trained) |

The output is a **keypoints file** containing X, Y coordinates and confidence scores for each body part, for every frame.

### Stage 3: Preprocessing

Raw pose data needs cleaning:

- **Interpolation**: Fills in brief tracking gaps when a body part is occluded
- **ID refinement**: Corrects animal identity swaps in multi-animal videos
- **Trimming**: Removes frames before/after the actual experiment

### Stage 4: Feature Extraction

From the cleaned pose data, behavysis calculates derived metrics:

- **Velocities** and **accelerations**
- **Distances** between body parts
- **Angles** (head direction, body curvature)
- **Distances** to arena features (center, walls, other animals)

These features become the input for behavior classification.

### Stage 5: Behavior Classification *(Optional)*

Machine learning models predict behaviors from the feature set:

```python
# Example: A "freezing" classifier might look for:
- Low body velocity
- Small bounding box area
- Sustained duration (>2 seconds)
```

You can:
- Use pre-trained classifiers
- Train your own with human-annotated data
- Adjust detection thresholds

### Stage 6: Quantitative Analysis

Finally, statistical analyses summarize behavior:

| Analysis | What It Measures |
|----------|------------------|
| **Thigmotaxis** | Time spent near walls (anxiety indicator) |
| **Speed** | Movement velocity over time |
| **Center Crossing** | Entries into/exits from center zone |
| **Freezing** | Duration of immobility |
| **Social Distance** | Distance between animals (for social experiments) |
| **In ROI** | Time spent in regions of interest |

---

## The project Folder Structure

behavysis uses a **numbered folder system** to organize data at each processing stage:

```
my_project/
├── 0_configs/           # JSON configuration files
│   └── experiment_name.json
├── 1_raw_vid/           # Your original videos
│   └── experiment_name.mp4
├── 2_formatted_vid/     # Resized, standardized videos
│   └── experiment_name.mp4
├── 3_keypoints/         # DLC pose output
│   └── experiment_name.parquet
├── 4_preprocessed/      # Cleaned pose data
│   └── experiment_name.parquet
├── 5_features_extracted/# Derived features
│   └── experiment_name.parquet
├── 6_predicted_behavs/  # ML behavior predictions
│   └── experiment_name.parquet
├── 7_scored_behavs/     # Human-verified behaviors
│   └── experiment_name.parquet
├── 8_analysis/          # Analysis results
│   ├── binned/          # Time-binned statistics
│   └── summary/         # Overall statistics
├── 9_analysis_combined/ # Combined across experiments
│   └── experiment_name.parquet
├── 10_evaluate_vid/     # Annotated evaluation videos
│   └── experiment_name.mp4
└── 0_diagnostics/       # Processing logs
    └── [process_name].csv
```

!!! important "Key Rule"
    All files for a single experiment must share the **same base name** (e.g., `mouse_A_01`). Only the folder and extension differ.

---

## Configuration Files

The `0_configs/experiment_name.json` file controls all processing parameters. It has three sections:

### `user` — Your Settings

Parameters you define, such as:

```json
{
  "user": {
    "format_vid": {
      "height_px": 540,
      "width_px": 960,
      "fps": 15
    },
    "analyse": {
      "thigmotaxis": {
        "thresh_mm": 50
      }
    }
  }
}
```

### `auto` — Calculated Values

Parameters computed automatically from your data:

- `start_frame`: When the animal first appears
- `stop_frame`: When the experiment ends
- `px_per_mm`: Pixel-to-millimeter conversion
- `fps`: Actual frame rate (measured)

### `ref` — Reusable References

Define once, reference anywhere. Use `--` prefix to reference:

```json
{
  "ref": {
    "bodyparts-centre": ["BodyCentre", "LeftFlankMid", "RightFlankMid"]
  },
  "user": {
    "analyse": {
      "speed": {
        "bodyparts": "--bodyparts-centre"
      }
    }
  }
}
```

!!! tip "Configuration Guide"
    See the [Configuration Tutorial](configs_json.md) for complete details on all parameters.

---

## Next Steps

Now you understand the workflow, you're ready to:

1. **[Set up your project folder](setup.md)** — Learn the exact file structure needed
2. **[Configure your parameters](configs_json.md)** — Understand all configuration options
3. **[Run your first analysis](../examples/analysis.md)** — Step-by-step guide to processing data

---

## Summary

| Concept | Description |
|---------|-------------|
| **Experiment** | A single video and its associated data files |
| **Project** | A folder containing multiple experiments |
| **Keypoints** | X,Y coordinates of tracked body parts |
| **Features** | Derived measurements from keypoints |
| **Behaviors** | Classified actions (freezing, grooming, etc.) |
| **Analysis** | Statistical summaries and plots |

behavysis handles the computational complexity so you can focus on the science.
