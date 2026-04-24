# behavysis Documentation

Welcome to the behavysis documentation — a complete guide for analyzing laboratory mouse behavior from video recordings.

## What is behavysis?

behavysis is a Python package that automates the conversion of raw video footage into quantitative behavioral data. It integrates with [DeepLabCut](http://www.mackenziemathislab.org/deeplabcut) for pose estimation and supports machine learning-based behavior classification.

### Key Features

- 🔍 **Pose Estimation**: Track body parts (nose, ears, tail, etc.) automatically
- 🎯 **Behavior Detection**: Identify behaviors like freezing, thigmotaxis, social interaction
- 📊 **Quantitative Analysis**: Generate statistics and time-series data
- 🎥 **Video Annotation**: Create evaluation videos with overlays
- 🖥️ **GUI Viewer**: Semi-automated behavior verification tool
- ⚡ **Parallel Processing**: Analyze multiple experiments efficiently

---

## Quick Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Getting Started**

    ---

    New to behavysis? Start here for installation and your first analysis.

    [:octicons-arrow-right-24: Installation](installation/installing.md)
    
    [:octicons-arrow-right-24: First Tutorial](tutorials/explanation.md)

-   :material-book-open-variant:{ .lg .middle } **Tutorials**

    ---

    Step-by-step guides to learn behavysis from scratch.

    [:octicons-arrow-right-24: Understanding the Workflow](tutorials/explanation.md)
    
    [:octicons-arrow-right-24: Project Setup](tutorials/setup.md)
    
    [:octicons-arrow-right-24: Configuration Files](tutorials/configs_json.md)

-   :material-tools:{ .lg .middle } **How-to Guides**

    ---

    Practical recipes for common tasks.

    [:octicons-arrow-right-24: Run Analysis](examples/analysis.md)
    
    [:octicons-arrow-right-24: Train Behavior Classifiers](examples/train.md)

-   :material-code-braces:{ .lg .middle } **API Reference**

    ---

    Detailed documentation of all classes and functions.

    [:octicons-arrow-right-24: Project API](reference/project.md)
    
    [:octicons-arrow-right-24: Experiment API](reference/experiment.md)
    
    [:octicons-arrow-right-24: All Processes](reference/processes.md)

</div>

---

## The Analysis Pipeline

behavysis follows a structured pipeline from raw video to results:

```mermaid
graph LR
    A[Raw Video] --> B[Format Video]
    B --> C[Pose Estimation<br/>(DeepLabCut)]
    C --> D[Preprocess Data]
    D --> E[Extract Features]
    E --> F[Classify Behaviors]
    F --> G[Quantitative Analysis]
    G --> H[Results & Plots]
```

Each step is configurable through JSON configuration files, allowing precise control over parameters.

---

## Who is behavysis for?

behavysis is designed for:

- 🧠 **Neuroscientists** studying mouse behavior
- 🔬 **Lab researchers** who need reproducible analysis
- 💻 **Non-programmers** who want an accessible Python interface
- 🤖 **Computational researchers** building behavior analysis pipelines

No advanced programming knowledge is required — basic Python familiarity is sufficient.

---

## Getting Help

!!! question "Need assistance?"

    - Check the [Troubleshooting](tutorials/diagnostics_messages.md) guide
    - Review [Error Messages](tutorials/diagnostics_messages.md#common-errors-and-warnings)
    - Open an issue on [GitHub](https://github.com/tlee08/behavysis/issues)

---

## Citation

If you use behavysis in your research, please cite both behavysis and the original DeepLabCut publication:

```bibtex
@software{behavysis,
  author = {BowenLab},
  title = {behavysis: Automated behavior analysis for laboratory mice},
  url = {https://github.com/tlee08/behavysis}
}

@article{mathis2018deeplabcut,
  title={DeepLabCut: markerless pose estimation of user-defined body parts with deep learning},
  author={Mathis, Alexander and Mamidanna, Pranav and Cury, Kevin M and others},
  journal={Nature Neuroscience},
  year={2018}
}
```
