"""Unit tests for template rendering utilities."""

from pathlib import Path

from behavysis.utils.template_utils import render_template


class TestRenderDlcSubproc:
    """Tests for dlc_subproc.py template rendering."""

    def test_renders_correctly(self) -> None:
        """dlc_subproc template renders with proper variable substitution."""
        code = render_template(
            "dlc/dlc_subproc.py",
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


class TestWhitespaceNormalization:
    """Tests for the whitespace normalization in render_template."""

    def _render(self, **kwargs) -> str:
        defaults = {
            "vid_fp_ls": [],
            "model_fp": Path("/tmp"),
            "temp_dlc_dir": Path("/tmp"),
            "gputouse": 0,
        }
        defaults.update(kwargs)
        return render_template("dlc/dlc_subproc.py", **defaults)

    def test_removes_trailing_blank_lines(self) -> None:
        """Rendered output has no trailing blank lines."""
        code = self._render()
        assert not code.rstrip("\n").endswith("\n\n")

    def test_no_leading_blank_lines(self) -> None:
        """Rendered output has no leading blank lines."""
        code = self._render()
        assert not code.startswith("\n")

    def test_no_consecutive_blank_lines(self) -> None:
        """No three or more consecutive newlines in output."""
        code = self._render()
        assert "\n\n\n" not in code
