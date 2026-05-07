"""
Tests for M11: stage_compare.py

Verifies delta computation, Markdown rendering, and file output using
synthetic metrics.json fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ods_phenocontext.schema import LABEL_NAMES
from ods_phenocontext.stage_compare import compare_stages, load_experiment_metrics, render_markdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(tmp_path: Path, stage: str, metrics: dict) -> Path:
    exp_dir = tmp_path / stage
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": "2026-05-07T00:00:00+00:00",
        "stage": stage,
        "eval_split": "val",
        "metrics": metrics,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(payload))
    return exp_dir


def _base_metrics(macro_f1: float = 0.5, f1_offset: float = 0.0) -> dict:
    m: dict = {
        "macro_f1": macro_f1,
        "micro_f1": macro_f1,
        "n_evaluated": 10,
        "abstention_rate": 0.0,
        "rule_id_counts": {},
        "source_coverage": {
            "rules": {"count": 10, "fraction": 1.0},
            "biobert": {"count": 0, "fraction": 0.0},
            "none": {"count": 0, "fraction": 0.0},
        },
    }
    for name in LABEL_NAMES:
        m[f"f1_{name}"] = round(macro_f1 + f1_offset, 3)
        m[f"precision_{name}"] = 0.5
        m[f"recall_{name}"] = 0.5
        m[f"tp_{name}"] = 3
        m[f"fp_{name}"] = 1
        m[f"fn_{name}"] = 1
        m[f"tn_{name}"] = 5
    return m


# ---------------------------------------------------------------------------
# load_experiment_metrics
# ---------------------------------------------------------------------------


class TestLoadExperimentMetrics:
    def test_loads_metrics(self, tmp_path: Path):
        exp_dir = _write_metrics(tmp_path, "rules_only", _base_metrics())
        data = load_experiment_metrics(exp_dir)
        assert "metrics" in data
        assert data["stage"] == "rules_only"

    def test_missing_metrics_json_raises(self, tmp_path: Path):
        exp_dir = tmp_path / "missing"
        exp_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_experiment_metrics(exp_dir)


# ---------------------------------------------------------------------------
# compare_stages
# ---------------------------------------------------------------------------


class TestCompareStages:
    def test_deltas_computed_relative_to_baseline(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics(macro_f1=0.5)),
            "biobert_only": _write_metrics(tmp_path, "biobert_only", _base_metrics(macro_f1=0.6)),
        }
        result = compare_stages(dirs, baseline_stage="rules_only")
        assert result["deltas"]["biobert_only"]["macro_f1"] == pytest.approx(0.1)

    def test_baseline_not_in_deltas(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics()),
            "biobert_only": _write_metrics(tmp_path, "biobert_only", _base_metrics()),
        }
        result = compare_stages(dirs, baseline_stage="rules_only")
        assert "rules_only" not in result["deltas"]

    def test_per_label_deltas_present(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics(macro_f1=0.5)),
            "combined": _write_metrics(tmp_path, "combined", _base_metrics(macro_f1=0.7)),
        }
        result = compare_stages(dirs, baseline_stage="rules_only")
        for label in LABEL_NAMES:
            assert f"f1_{label}" in result["deltas"]["combined"]

    def test_coverage_extracted(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics()),
        }
        result = compare_stages(dirs, baseline_stage="rules_only")
        assert "rules" in result["coverage"]["rules_only"]

    def test_invalid_baseline_raises(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics()),
        }
        with pytest.raises(ValueError, match="baseline_stage"):
            compare_stages(dirs, baseline_stage="nonexistent")

    def test_negative_delta(self, tmp_path: Path):
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics(macro_f1=0.8)),
            "biobert_only": _write_metrics(tmp_path, "biobert_only", _base_metrics(macro_f1=0.5)),
        }
        result = compare_stages(dirs, baseline_stage="rules_only")
        assert result["deltas"]["biobert_only"]["macro_f1"] == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def _make_comparison(self, tmp_path: Path) -> tuple[dict, dict[str, Path]]:
        dirs = {
            "rules_only": _write_metrics(tmp_path, "rules_only", _base_metrics(macro_f1=0.5)),
            "biobert_only": _write_metrics(tmp_path, "biobert_only", _base_metrics(macro_f1=0.6)),
            "combined": _write_metrics(tmp_path, "combined", _base_metrics(macro_f1=0.7)),
        }
        comparison = compare_stages(dirs, baseline_stage="rules_only")
        return comparison, dirs

    def test_returns_nonempty_string(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        assert len(md) > 0

    def test_contains_label_names(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        for label in LABEL_NAMES:
            assert label in md

    def test_contains_stage_names(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        for stage in dirs:
            assert stage in md

    def test_contains_delta_signs(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        assert "+" in md or "-" in md

    def test_writes_file_when_output_path_given(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        out = tmp_path / "report" / "comparison.md"
        render_markdown(comparison, "rules_only", dirs, output_path=out)
        assert out.exists()
        assert len(out.read_text()) > 0

    def test_creates_parent_dirs(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        out = tmp_path / "a" / "b" / "c" / "report.md"
        render_markdown(comparison, "rules_only", dirs, output_path=out)
        assert out.exists()

    def test_macro_f1_present_in_output(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        assert "macro_f1" in md

    def test_source_coverage_section_present(self, tmp_path: Path):
        comparison, dirs = self._make_comparison(tmp_path)
        md = render_markdown(comparison, "rules_only", dirs)
        assert "Source Coverage" in md
