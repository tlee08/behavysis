# Social Interaction — Two Mice

Social interaction experiment with one marked and one unmarked mouse.

## 1. Create the project

```bash
behavysis-make-project --preset social_two_mice
```

## 2. Config differences from open field

The social preset adds:

- **`preprocess.refine_ids`** — swaps identities when the ID tracker flips
  animals. Set `marked`/`unmarked` to match your DLC model's individual names.
  Set `marking` to the bodypart that distinguishes marked from unmarked (e.g.
  `AnimalColourMark`).

- **`analyse.social_distance`** — distance between the two animals' centre
  points over time.

- **`extract_features`** uses `*indivs_simba` (two individuals) instead of
  `*indivs_single`.

## 3. Identity refinement

If your DLC model produces multi-animal keypoints but mixes up identities (e.g.
mouse1 and mouse2 swap), enable `refine_ids` in the config. It uses the marking
bodypart to correct identities:

```yaml
preprocess:
  refine_ids:
    marked: mouse1marked
    unmarked: mouse2unmarked
    marking: AnimalColourMark
    window_sec: 0.5
    bodyparts: *bpts_centre
```

If you don't need identity refinement (e.g. your DLC model handles it), remove
`refine_ids` from the config. Then remove the import in `run_pipeline.py` and
drop it from the `preprocess(funcs=...)` call.

## 4. Run

```bash
marimo edit run_pipeline.py
```

Same flow as open field. The preprocess stage now includes `refine_ids`, and the
analyse stage includes `social_distance`.

## 5. Results

Additional output in `8_analysis/social_distance/`:

- Distance between animals over time (frame-by-frame)
- Time-binned summaries
- Cross-experiment collated parquet
