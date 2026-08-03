# behavysis 🐭

[![Documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg)](https://tlee08.github.io/behavysis/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)](LICENSE)

**Automated behavior analysis for laboratory mice from video recordings.**

behavysis is a Python package that transforms raw video footage of lab mice into quantitative behavioral data. It combines pose estimation (via DeepLabCut), feature extraction, and machine learning to detect and analyze behaviors like thigmotaxis (wall-hugging), freezing, social interactions, and more.

---

## What behavysis Does

Working with laboratory mice videos? behavysis automates the entire analysis pipeline:

1. **Video Formatting** — Standardizes video files (resolution, frame rate)
2. **Pose Estimation** — Detects body parts (nose, ears, tail, etc.) using DeepLabCut
3. **Data Preprocessing** — Cleans and interpolates tracking data
4. **Behavior Detection** — Identifies behaviors using trained classifiers
5. **Quantitative Analysis** — Calculates metrics like speed, distance traveled, time in regions

### Example Outputs

- Per-frame behavior annotations
- Summary statistics (mean speed, time in center, etc.)
- Time-binned analysis for temporal patterns
- Annotated evaluation videos

---

## Installation

### Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Linux/Mac/Windows** with 8GB+ RAM recommended
- **GPU** (optional but recommended for pose estimation)

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/tlee08/behavysis.git
cd behavysis

# 2. Create the conda environment
conda env create -f conda_env.yaml
conda activate behavysis

# 3. Install the package
pip install -e .
```

### Additional Dependencies

For pose estimation and behavior classification, you'll also need:

```bash
# DeepLabCut environment (for running DLC models)
conda env create -f https://raw.githubusercontent.com/DeepLabCut/DeepLabCut/main/conda-environments/DEEPLABCUT.yaml

# SimBA environment (optional, for behavior classification)
conda env create -f simba_env.yaml
```

📚 **See the [full installation guide](https://tlee08.github.io/behavysis/installation/installing/) for detailed instructions.**

---

## Quick Start

### 1. Project Setup

Organize your project folder like this:

```
my_project/
├── 0_configs/          # Configuration files (auto-generated)
├── 1_raw_vid/          # Your raw video files (.mp4)
├── 2_formatted_vid/    # Formatted videos (auto-generated)
├── 3_keypoints/        # DLC pose data (auto-generated)
├── 4_preprocessed/     # Cleaned data (auto-generated)
│   └── ...
└── 9_analysis_combined/ # Final analysis (auto-generated)
```

### 2. Running Analysis

```python
from behavysis import Project

# Initialize project
proj = Project("./my_project")

# Import experiments (videos in 1_raw_vid/)
proj.import_experiments()

# Format videos (resize, adjust fps)
proj.format_vid(overwrite=True)

# Run pose estimation with DeepLabCut
proj.run_dlc(gputouse=0, overwrite=True)

# Calculate parameters and preprocess
proj.calculate_parameters((
    CalculateParams.start_frame,
    CalculateParams.stop_frame,
    CalculateParams.px_per_mm,
))
proj.preprocess((
    Preprocess.start_stop_trim,
    Preprocess.interpolate,
), overwrite=True)

# Extract features for behavior classification
proj.extract_features(overwrite=True)

# Run analysis
proj.analyse((
    Analyse.thigmotaxis,      # Wall-hugging behavior
    Analyse.speed,             # Movement speed
    Analyse.center_crossing,   # Center zone entries
    Analyse.freezing,          # Freezing behavior
))
```

📚 **See the [tutorial](https://tlee08.github.io/behavysis/tutorials/explanation/) for the complete workflow.**

---

## Key Concepts for New Users

| Term | Description |
|------|-------------|
| **Experiment** | A single video recording and its associated data |
| **Project** | A collection of experiments processed together |
| **Keypoints** | Tracked body parts (nose, ears, body center, etc.) |
| **Configs** | JSON files that control all processing parameters |
| **Features** | Derived measurements (speed, distances, angles) |
| **Scored Behaviors** | Human-verified behavior annotations for training |

---

## Documentation

The documentation is organized following the [Diátaxis](https://diataxis.fr/) framework:

| Section | Purpose |
|---------|---------|
| **[Tutorials](https://tlee08.github.io/behavysis/tutorials/setup/)** | Step-by-step learning for newcomers |
| **[How-to Guides](https://tlee08.github.io/behavysis/examples/analysis/)** | Task-oriented recipes |
| **[Reference](https://tlee08.github.io/behavysis/reference/behavysis/)** | API documentation |
| **[Installation](https://tlee08.github.io/behavysis/installation/installing/)** | Setup and configuration |

---

## Citation

If you use behavysis in your research, please cite:

```bibtex
@software{behavysis,
  author = {BowenLab},
  title = {behavysis: Automated behavior analysis for laboratory mice},
  url = {https://github.com/tlee08/behavysis}
}
```

And the foundational work this package builds upon:

- **DeepLabCut**: Mathis et al. (2018) *Nature Neuroscience*
- **SimBA**: Nilsson et al. — [github.com/sgoldenlab/simba](https://github.com/sgoldenlab/simba)

---

## Contributing

Contributions are welcome! Please see our [GitHub repository](https://github.com/tlee08/behavysis) to:

- Report issues
- Suggest features
- Submit pull requests

---

## License

behavysis is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.

---

## Support

- 📖 **Documentation**: [tlee08.github.io/behavysis](https://tlee08.github.io/behavysis/)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tlee08/behavysis/issues)
- 💬 **Questions**: Open a discussion on GitHub
