# Preset Reference

Every preset ships as a `default_config.yaml` + `run_pipeline.py` notebook.
The `base` preset below documents every config field — it's the canonical
reference. Other presets are subsets for common experiment types.

```bash
behavysis-make-project --list      # see available presets
behavysis-make-project --preset base  # scaffold the reference
```

## base — Full Reference

```yaml
--8<-- "base/default_config.yaml"
```
