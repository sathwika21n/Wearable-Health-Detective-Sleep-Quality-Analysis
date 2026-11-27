"""Tests for the one-command runner in run.py."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run  # noqa: E402


def write(path: Path, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A tiny pipeline: raw -> aligned -> results (dashboard) and -> features (analysis)."""
    monkeypatch.setattr(run, "PROJECT_DIR", tmp_path)
    for script in ("align.py", "dashboard.py", "features.py"):
        write(tmp_path / "scripts" / script, 50)
    raw, aligned = tmp_path / "raw", tmp_path / "aligned"
    results, features = tmp_path / "results.csv", tmp_path / "features.csv"
    # The raw data looks newer than the aligned data, so the raw step is outdated,
    # while the dashboard table looks up to date compared with the aligned data.
    write(raw / "p1.csv", 250)
    write(aligned / "p1_aligned.csv", 200)
    write(results, 300)
    write(features, 300)
    steps = [
        run.Step("Align", "align.py", inputs=[raw], outputs=[aligned], group="raw"),
        run.Step("Dashboard table", "dashboard.py", inputs=[aligned], outputs=[results], group="dashboard"),
        run.Step("Features", "features.py", inputs=[aligned], outputs=[features], group="analysis"),
    ]
    return steps, aligned


def args(*flags):
    return run.build_parser().parse_args(list(flags))


def test_from_raw_reruns_the_dashboard_table_after_realigning(project):
    steps, aligned = project
    ran = []

    def runner(step):
        ran.append(step.name)
        if step.name == "Align":
            write(aligned / "p1_aligned.csv", 400)  # rebuilt, now newer than the results

    run.run_pipeline(steps, args("--from-raw"), runner=runner)
    assert ran == ["Align", "Dashboard table"]


def test_downstream_step_runs_even_if_timestamps_do_not_move(project):
    steps, _aligned = project
    ran = []
    # The runner leaves every timestamp alone (e.g. a coarse file system clock),
    # so only the record of what was rebuilt in this run can trigger the rerun.
    run.run_pipeline(steps, args("--from-raw"), runner=lambda step: ran.append(step.name))
    assert ran == ["Align", "Dashboard table"]


def test_unselected_analysis_step_is_left_as_committed(project, capsys):
    steps, _aligned = project
    ran = []
    run.run_pipeline(steps, args("--from-raw"), runner=lambda step: ran.append(step.name))
    assert "Features" not in ran
    assert "Add --all" in capsys.readouterr().out


def test_all_also_rebuilds_downstream_analysis(project):
    steps, _aligned = project
    ran = []
    run.run_pipeline(steps, args("--from-raw", "--all"), runner=lambda step: ran.append(step.name))
    assert ran == ["Align", "Dashboard table", "Features"]


def test_nothing_runs_when_everything_is_up_to_date(project):
    steps, _aligned = project
    ran = []
    run.run_pipeline(steps, args(), runner=lambda step: ran.append(step.name))
    assert ran == []


def test_plan_preview_matches_the_run(project):
    steps, _aligned = project
    plan = run.plan_steps(steps, args("--from-raw"))
    assert [(step.name, will_run) for step, _status, will_run in plan] == [
        ("Align", True), ("Dashboard table", True), ("Features", False),
    ]


@pytest.mark.parametrize("code", [0, 1, 3])
def test_launch_dashboard_returns_streamlits_exit_code(monkeypatch, code):
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, code))
    assert run.launch_dashboard() == code


def test_launch_dashboard_treats_ctrl_c_as_a_normal_stop(monkeypatch):
    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(run.subprocess, "run", interrupted)
    assert run.launch_dashboard() == 0


def test_main_exits_with_the_dashboards_exit_code(project, monkeypatch):
    steps, _aligned = project
    monkeypatch.setattr(run, "STEPS", steps)
    monkeypatch.setattr(run, "RAW_DIR", run.PROJECT_DIR / "raw")
    monkeypatch.setattr(run, "check_requirements", lambda need_dashboard: None)
    monkeypatch.setattr(run, "launch_dashboard", lambda: 1)
    assert run.main([]) == 1
