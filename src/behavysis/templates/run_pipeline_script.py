# Auto-generated Behavysis pipeline script
# Regenerate from the marimo notebook to update.
from pathlib import Path

from behavysis import Project
{% if func_imports %}
from behavysis.funcs import (
{%- for fname in func_imports | sort %}
    {{ fname }},
{%- endfor %}
)
{% endif %}

names_ls = [i.name for i in (Path({{ project_fp_repr }}) / "1_raw_videos").iterdir()]
proj = Project(Path({{ project_fp_repr }}))
proj.nprocs = {{ nprocs }}
proj.import_experiments()
{% if update_config %}
proj.update_config(default_config_fp=Path({{ config_fp_repr }}), overwrite='user')
{% endif %}
{% if format_vid %}
proj.format_video(overwrite={{ overwrite }})
{% endif %}
{% if run_dlc %}
proj.run_dlc(gputouse=None, overwrite={{ overwrite }})
{% endif %}
{% if calculate_parameters %}
proj.calculate_parameters(funcs=({{ calc_funcs | join(', ') }},))
{% endif %}
{% if preprocess %}
proj.preprocess(funcs=({{ prep_funcs | join(', ') }},), overwrite={{ overwrite }})
{% endif %}
{% if analyse %}
proj.analyse(funcs=({{ anal_funcs | join(', ') }},))
proj.combine_analysis()
proj.collate_analysis()
{% endif %}
{% if extract_features %}
proj.extract_features(overwrite={{ overwrite }})
{% endif %}
{% if classify_behaviour %}
proj.classify_behaviour(overwrite={{ overwrite }})
proj.export_behaviour(overwrite={{ overwrite }})
{% endif %}
{% if analyse_behaviour %}
proj.analyse_behaviour()
{% endif %}
{% if combine_analysis %}
proj.combine_analysis()
proj.collate_analysis()
{% endif %}
