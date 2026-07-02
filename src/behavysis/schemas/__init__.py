"""Schemas."""

# TODO:
# 🟠 MAJOR: schemas/ has an identity crisis
# schemas/
# ├── schemas.py          ← Schema dicts + read_df/write_df  (182 lines) ✅
# ├── analysis_agg.py     ← agg_quantitative, make_binned, summary_binned (348 lines) ❌ NOT a schema
# ├── behaviour.py        ← vect2bouts, bouts2frames, predicted2scored (362 lines) ❌ NOT a schema
# └── keypoints.py        ← check_bpts_exist, get_indivs_bpts (49 lines)    ❌ NOT a schema
# 710 of 941 lines in the schemas/ package are domain transformation functions, not schemas. from behavysis.schemas import vect2bouts is semantically wrong — the caller is importing business logic from what it believes is a schema definitions module.
# The fix:
# - schemas.py stays as schemas/ (schema dicts + I/O validation at the boundary)
# - behaviour.py → funcs/behaviour_transforms.py (bout detection, frame↔bout conversion, BORIS import)
# - analysis_agg.py → funcs/analysis_transforms.py or merge into funcs/analyse/
# - keypoints.py → funcs/keypoint_utils.py

from .analysis_agg import (
    agg_behaviour,
    agg_quantitative,
    make_binned,
    make_binned_plot,
    summary_binned,
    summary_binned_behaviour,
    summary_binned_quantitative,
)
from .behaviour import (
    bouts2frames,
    frames2bouts,
    get_bouts_struct,
    import_boris_tsv,
    merge_bouts,
    predicted2scored,
    vect2bouts,
)
from .keypoints import check_bpts_exist, get_indivs_bpts
from .schemas import (
    ANALYSIS_SCHEMA,
    BEHAVIOUR_PREDICTED_SCHEMA,
    BEHAVIOUR_SCORED_BASE,
    BINNED_SCHEMA,
    COLLATED_BINNED_SCHEMA,
    COLLATED_SUMMARY_SCHEMA,
    COMBINED_ANALYSIS_SCHEMA,
    FEATURES_BASE,
    KEYPOINTS_SCHEMA,
    SUMMARY_SCHEMA,
    SchemaDict,
    init_empty_df,
    read_csv,
    read_df,
    write_csv,
    write_df,
)

__all__ = [
    "ANALYSIS_SCHEMA",
    "BEHAVIOUR_PREDICTED_SCHEMA",
    "BEHAVIOUR_SCORED_BASE",
    "BINNED_SCHEMA",
    "COLLATED_BINNED_SCHEMA",
    "COLLATED_SUMMARY_SCHEMA",
    "COMBINED_ANALYSIS_SCHEMA",
    "FEATURES_BASE",
    "KEYPOINTS_SCHEMA",
    "SUMMARY_SCHEMA",
    "SchemaDict",
    "agg_behaviour",
    "agg_quantitative",
    "bouts2frames",
    "check_bpts_exist",
    "frames2bouts",
    "get_bouts_struct",
    "get_indivs_bpts",
    "import_boris_tsv",
    "init_empty_df",
    "make_binned",
    "make_binned_plot",
    "merge_bouts",
    "predicted2scored",
    "read_csv",
    "read_df",
    "summary_binned",
    "summary_binned_behaviour",
    "summary_binned_quantitative",
    "vect2bouts",
    "write_csv",
    "write_df",
]
