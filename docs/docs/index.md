# Behavysis

Animal behaviour video-analysis pipeline from raw footage to publication-ready
results.

## Quick start

```bash
conda env create -f conda_env.yaml    # install behaviours
conda activate behavysis
behavysis-init                         # install DEEPLABCUT environment
behavysis-make-project                 # scaffold a project
# → Edit default_config.yaml
# → Copy .mp4 videos into 1_raw_videos/
marimo edit run_pipeline.py            # run the pipeline
```

## Using the documentation

<div class="grid cards" markdown>

- :material-school: **[Tutorials](tutorials/install.md)**

    ---

    Learn by doing. Step-by-step guides that take you from install to your
    first completed analysis.

- :material-head-question: **[How-to guides](how-to/open-field.md)**

    ---

    Solve specific problems. How to set up an open field experiment, how to
    train a classifier, how to debug common errors.

- :material-bookshelf: **[Reference](reference/behavysis.md)**

    ---

    Technical API documentation generated from source code. Config schema,
    function signatures, and class APIs.

- :material-lightbulb: **[Explanation](explanation/pipeline.md)**

    ---

    Background and context. How the pipeline is architected, how the config
    system works, and why things are designed the way they are.

</div>
