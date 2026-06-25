"""Unit tests for template rendering utilities."""

import ast
from pathlib import Path

from behavysis.utils.template_utils import render_template, save_template

_PIPELINE_PARAMS = dict(
    project_fp_repr=repr("/Users/test/project"),
    config_fp_repr=repr("/Users/test/config.json"),
    nprocs=5,
    overwrite=False,
)


def _render_pipeline(**overrides) -> str:
    kwargs: dict = {
        "update_config": False,
        "format_vid": False,
        "run_dlc": False,
        "calculate_parameters": False,
        "preprocess": False,
        "analyse": False,
        "extract_features": False,
        "classify_behaviour": False,
        "analyse_behaviour": False,
        "combine_analysis": False,
        "calc_funcs": [],
        "prep_funcs": [],
        "anal_funcs": [],
        "func_imports": set(),
        **_PIPELINE_PARAMS,
    }
    kwargs.update(overrides)
    # Mirror notebook Cell behaviour: step can't be on with zero funcs
    if not kwargs["calc_funcs"]:
        kwargs["calculate_parameters"] = False
    if not kwargs["prep_funcs"]:
        kwargs["preprocess"] = False
    if not kwargs["anal_funcs"]:
        kwargs["analyse"] = False
    return render_template("run_pipeline_script.py", **kwargs)


class TestRenderPipelineScript:
    """Tests for run_pipeline_script.py template rendering."""

    def test_all_checkboxes_off(self) -> None:
        """Only imports and project init when all checkboxes are off."""
        code = _render_pipeline()

        assert "from behavysis import Project" in code
        assert "proj = Project" in code
        assert "proj.import_experiments()" in code
        assert "update_config" not in code
        assert "format_video" not in code
        assert "run_dlc" not in code
        assert "calculate_parameters" not in code
        assert "preprocess" not in code
        assert "analyse" not in code
        assert "extract_features" not in code
        assert "classify_behaviour" not in code
        assert "export_behaviour" not in code
        assert "analyse_behaviour" not in code

    def test_all_checkboxes_on(self) -> None:
        """All pipeline steps present when all checkboxes are on."""
        code = _render_pipeline(
            update_config=True,
            format_vid=True,
            run_dlc=True,
            calculate_parameters=True,
            preprocess=True,
            analyse=True,
            extract_features=True,
            classify_behaviour=True,
            analyse_behaviour=True,
            combine_analysis=True,
            calc_funcs=["start_frame_from_likelihood", "stop_frame_from_dur"],
            prep_funcs=["interpolate", "start_stop_trim"],
            anal_funcs=["speed"],
            func_imports={
                "start_frame_from_likelihood",
                "stop_frame_from_dur",
                "interpolate",
                "start_stop_trim",
                "speed",
            },
        )

        assert "proj.update_config" in code
        assert "proj.format_video" in code
        assert "proj.run_dlc" in code
        assert "proj.calculate_parameters" in code
        assert "proj.preprocess" in code
        assert "proj.analyse" in code
        assert "proj.extract_features" in code
        assert "proj.classify_behaviour" in code
        assert "proj.export_behaviour" in code
        assert "proj.analyse_behaviour" in code

    def test_func_imports_only_when_funcs_selected(self) -> None:
        """Func imports present only when functions are selected."""
        code_with = _render_pipeline(
            calculate_parameters=True,
            calc_funcs=["start_frame_from_likelihood"],
            func_imports={"start_frame_from_likelihood"},
        )
        assert "from behavysis.funcs import (" in code_with
        assert "start_frame_from_likelihood" in code_with

        code_without = _render_pipeline()
        assert "from behavysis.funcs import (" not in code_without

    def test_missing_funcs_skips_block(self) -> None:
        """Checkbox on but no funcs selected omits the step entirely."""
        code = _render_pipeline(
            calculate_parameters=True,
            calc_funcs=[],
            func_imports=set(),
        )
        assert "proj.calculate_parameters" not in code

    def test_generated_script_is_valid_python(self) -> None:
        """Generated output is syntactically valid Python."""
        code = _render_pipeline(
            update_config=True,
            format_vid=True,
            run_dlc=True,
            calculate_parameters=True,
            preprocess=True,
            analyse=True,
            extract_features=True,
            classify_behaviour=True,
            analyse_behaviour=True,
            combine_analysis=True,
            calc_funcs=[
                "start_frame_from_likelihood",
                "stop_frame_from_dur",
                "dur_frames_from_likelihood",
                "px_per_mm",
            ],
            prep_funcs=["start_stop_trim", "interpolate"],
            anal_funcs=["in_roi", "speed", "distance"],
            func_imports={
                "start_frame_from_likelihood",
                "stop_frame_from_dur",
                "dur_frames_from_likelihood",
                "px_per_mm",
                "start_stop_trim",
                "interpolate",
                "in_roi",
                "speed",
                "distance",
            },
        )
        ast.parse(code)

    def test_starts_with_header_comment(self) -> None:
        """Generated script begins with the standard header."""
        code = _render_pipeline()
        assert code.startswith("# Auto-generated Behavysis pipeline script")

    def test_includes_project_init(self) -> None:
        """Project initialization always present."""
        code = _render_pipeline()
        assert "proj = Project(Path(" in code
        assert "proj.nprocs = 5" in code

    def test_single_blank_line_between_sections(self) -> None:
        """No more than one consecutive blank line between logical sections."""
        code = _render_pipeline(
            update_config=True,
            format_vid=True,
            run_dlc=True,
            calculate_parameters=True,
            calc_funcs=["start_frame_from_likelihood"],
            func_imports={"start_frame_from_likelihood"},
        )
        assert "\n\n\n" not in code

    def test_ends_with_single_newline(self) -> None:
        """Script ends with exactly one trailing newline."""
        code = _render_pipeline()
        assert code.endswith("\n")
        assert not code.endswith("\n\n")


class TestPathEscaping:
    """Tests for path escaping in generated scripts."""

    def test_windows_backslash_paths(self) -> None:
        """Windows UNC paths with backslashes produce valid Python."""
        code = _render_pipeline(
            project_fp_repr=repr(r"\\server\share\project"),
        )
        assert "server" in code
        assert "share" in code
        assert "project" in code
        ast.parse(code)

    def test_single_quote_in_path(self) -> None:
        """Paths containing single quotes are escaped safely."""
        code = _render_pipeline(
            project_fp_repr=repr("/Users/O'Brien/project"),
        )
        assert "O'Brien" in code
        ast.parse(code)


class TestRenderDlcSubproc:
    """Tests for dlc_subproc.py template rendering."""

    def test_renders_correctly(self) -> None:
        """dlc_subproc template renders with proper variable substitution."""
        code = render_template(
            "dlc_subproc.py",
            vid_fp_ls=["/path/to/vid1.mp4", "/path/to/vid2.mp4"],
            model_fp=Path("/models/config.yaml"),
            temp_dlc_dir=Path("/tmp/dlc"),
            gputouse=0,
        )

        assert "/path/to/vid1.mp4" in code
        assert "/path/to/vid2.mp4" in code
        assert 'r"/models/config.yaml"' in code
        assert 'r"/tmp/dlc"' in code
        assert "gputouse=0" in code


class TestSaveTemplate:
    """Tests for save_template disk writing."""

    def test_writes_to_disk(self, tmp_path: Path) -> None:
        """save_template writes rendered content to the specified path."""
        dst = tmp_path / "output.py"
        save_template(
            "run_pipeline_script.py",
            dst,
            **_PIPELINE_PARAMS,
            update_config=False,
            format_vid=False,
            run_dlc=False,
            calculate_parameters=False,
            preprocess=False,
            analyse=False,
            extract_features=False,
            classify_behaviour=False,
            analyse_behaviour=False,
            combine_analysis=False,
            calc_funcs=[],
            prep_funcs=[],
            anal_funcs=[],
            func_imports=set(),
        )

        assert dst.is_file()
        content = dst.read_text()
        assert "from behavysis import Project" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """save_template creates parent directories if they don't exist."""
        dst = tmp_path / "nested" / "dir" / "output.py"
        save_template(
            "run_pipeline_script.py",
            dst,
            **_PIPELINE_PARAMS,
            update_config=False,
            format_vid=False,
            run_dlc=False,
            calculate_parameters=False,
            preprocess=False,
            analyse=False,
            extract_features=False,
            classify_behaviour=False,
            analyse_behaviour=False,
            combine_analysis=False,
            calc_funcs=[],
            prep_funcs=[],
            anal_funcs=[],
            func_imports=set(),
        )

        assert dst.is_file()


class TestWhitespaceNormalization:
    """Tests for the whitespace normalization in render_template."""

    def test_removes_trailing_blank_lines(self) -> None:
        """Rendered output has no trailing blank lines."""
        code = _render_pipeline()
        assert not code.rstrip("\n").endswith("\n\n")

    def test_no_leading_blank_lines(self) -> None:
        """Rendered output has no leading blank lines."""
        code = _render_pipeline()
        assert not code.startswith("\n")

    def test_no_consecutive_blank_lines(self) -> None:
        """No three or more consecutive newlines in output."""
        code = _render_pipeline(
            update_config=True,
            format_vid=True,
            run_dlc=True,
            calculate_parameters=True,
            preprocess=True,
            analyse=True,
            extract_features=True,
            classify_behaviour=True,
            analyse_behaviour=True,
            combine_analysis=True,
            calc_funcs=["start_frame_from_likelihood"],
            prep_funcs=["interpolate"],
            anal_funcs=["speed"],
            func_imports={"start_frame_from_likelihood", "interpolate", "speed"},
        )
        assert "\n\n\n" not in code
