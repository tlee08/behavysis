# behavysis

Animal behaviour video-analysis pipeline: raw footage → DLC keypoint tracking →
quantitative analysis → behaviour classification → combined summary.

[Documentation](https://tlee08.github.io/behavysis/)

## Quick start

```bash
conda env create -f conda_env.yaml
conda activate behavysis
behavysis-init                       # install DEEPLABCUT environment
behavysis-make-project               # scaffold a project (pick a preset)
# Edit default_config.yaml
# Copy .mp4 videos into 1_raw_videos/
marimo edit run_pipeline.py          # run the pipeline
```

See the [full installation guide](https://tlee08.github.io/behavysis/installation/installing/)

## Developer setup

```bash
conda env create -f conda_env.yaml
conda activate behavysis
uv pip install -e ".[dev]"
```

See [AGENTS.md](AGENTS.md) for dev workflow.

## References

Mathis, A., et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nature Neuroscience*. [doi:10.1038/s41593-018-0209-y](http://doi.org/10.1038/s41593-018-0209-y)

Nath, T., et al. (2019). Using DeepLabCut for 3D markerless pose estimation across species and behaviours. *Nature Protocols*. [doi:10.1038/s41596-019-0176-0](http://doi.org/10.1038/s41596-019-0176-0)

Lauer, J., et al. (2022). Multi-animal pose estimation, identification and tracking with DeepLabCut. *Nature Methods*. [doi:10.1038/s41592-022-01443-0](http://doi.org/10.1038/s41592-022-01443-0)

Nilsson, S., et al. Simple Behavioural Analysis (SimBA). [github.com/sgoldenlab/simba](https://github.com/sgoldenlab/simba)
