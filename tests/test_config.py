"""Config loading utilities."""

from __future__ import annotations

import pytest

from upf_digital_twin.utils.config import (
    _find_project_root,
    load_configs,
    load_yaml,
    resolve_path,
)


def test_load_yaml_roundtrip(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nb:\n  c: two\n")
    assert load_yaml(p) == {"a": 1, "b": {"c": "two"}}


def test_find_project_root_walks_up_to_configs(tmp_path):
    (tmp_path / "configs").mkdir()
    nested = tmp_path / "src" / "pkg" / "deep"
    nested.mkdir(parents=True)
    assert _find_project_root(nested) == tmp_path


def test_find_project_root_raises_without_configs(tmp_path):
    with pytest.raises(FileNotFoundError):
        _find_project_root(tmp_path)


def test_load_configs_reads_both_files(tmp_path):
    cfg = tmp_path / "configs"
    cfg.mkdir()
    (cfg / "paths.yaml").write_text("results:\n  dir: results/x\n")
    (cfg / "scenario.yaml").write_text("traffic:\n  selected_k: 3\n")
    paths_cfg, scenario_cfg = load_configs(tmp_path)
    assert paths_cfg["results"]["dir"] == "results/x"
    assert scenario_cfg["traffic"]["selected_k"] == 3


def test_resolve_path_joins_relative(tmp_path):
    assert resolve_path(tmp_path, "a/b.txt") == tmp_path / "a/b.txt"
