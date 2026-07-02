# Troubleshoot Common Errors

## "Project folder not found"

Run `behavysis-make-project` first. The `Project` constructor expects the folder
structure created by the setup command.

## "DLC model config not found"

Check `run_dlc.model_fp` in `default_config.yaml`. It must be an absolute path:

```yaml
run_dlc:
  model_fp: /home/user/models/my_dlc/config.yaml
```

## "No files named X found"

Each experiment's files must share the same name (without extension) across all
stage folders. E.g. `exp1.mp4` in `1_raw_videos/` produces `exp1.parquet` in
`3_keypoints/`.

## "Bodyparts not found in keypoints data"

Your DLC model's bodypart names don't match the config. Check the anchor
definitions at the top of `default_config.yaml` and update them to match your
model's output. Run DLC's `check_labels` to see your model's bodypart names.

## "Width and height must be provided"

Run `format_video` stage first — it writes video metadata that downstream stages
need.

## "Field required" validation errors

Every top-level key in `default_config.yaml` is required (even if `null`). If
Pydantic complains about a missing field, add it explicitly:

```yaml
extract_features: null
classify_behaviour: null
```
