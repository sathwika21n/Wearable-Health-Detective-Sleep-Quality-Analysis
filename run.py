"""One entry point for the Sleep Detective project.

Run from the project root:

    python run.py                 build what the dashboard needs, then open it
    python run.py --check         only list each step and whether it is up to date
    python run.py --no-dashboard  build what the dashboard needs, don't open it
    python run.py --all           also rebuild the analysis tables in outputs/
    python run.py --from-raw      also re-align the raw data in data/ifh_affect (slow)
    python run.py --force         rerun the selected steps even if they look up to date

Steps run in pipeline order. A step runs when one of its outputs is missing or
older than one of its inputs. Each step is checked again when it is reached,
so a step whose inputs an earlier step just rebuilt runs too. Files that are
committed to the repo (the aligned data and the analysis tables) are only
rebuilt when --from-raw or --all asks for it, so a normal run never changes
tracked files.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "ifh_affect"
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
OUTPUTS = PROJECT_DIR / "outputs"
AWAKENINGS = OUTPUTS / "participant_nightly_awakening" / "nightly_awakenings.csv"
DASHBOARD_APP = PROJECT_DIR / "frontend" / "app.py"


@dataclass
class Step:
    name: str
    script: str
    inputs: list[Path]
    outputs: list[Path]
    group: str  # "raw", "dashboard" or "analysis"


STEPS = [
    Step(
        "Align raw data into 5-minute rows",
        "explore_data.py",
        inputs=[RAW_DIR],
        outputs=[ALIGNED_DIR, AWAKENINGS],
        group="raw",
    ),
    Step(
        "Build the dashboard's change-detection table",
        "build_change_detection_table.py",
        inputs=[ALIGNED_DIR, AWAKENINGS],
        outputs=[OUTPUTS / "change_detection_results.csv"],
        group="dashboard",
    ),
    Step(
        "Nightly sleep features",
        "nightly_sleep_features.py",
        inputs=[ALIGNED_DIR],
        outputs=[OUTPUTS / "nightly_sleep_features.csv"],
        group="analysis",
    ),
    Step(
        "Sleep-score model dataset",
        "build_sleep_score_dataset.py",
        inputs=[RAW_DIR],
        outputs=[OUTPUTS / "model_ready_sleep_score.csv"],
        group="analysis",
    ),
    Step(
        "Prior-10-night participant baselines",
        "participant_baselines.py",
        inputs=[OUTPUTS / "nightly_sleep_features.csv", OUTPUTS / "model_ready_sleep_score.csv"],
        outputs=[OUTPUTS / "participant_baselines.csv"],
        group="analysis",
    ),
    Step(
        "Heart-rate windows before each awakening",
        "awakening_hr_windows.py",
        inputs=[ALIGNED_DIR],
        outputs=[OUTPUTS / "awakening_hr_windows.csv"],
        group="analysis",
    ),
]


def _files(path: Path) -> list[Path]:
    if path.is_dir():
        return [item for item in path.rglob("*") if item.is_file()]
    return [path] if path.exists() else []


def step_status(step: Step) -> str:
    """Return "missing", "outdated" or "up to date" for one step."""
    output_files = [_files(path) for path in step.outputs]
    if any(not files for files in output_files):
        return "missing"
    input_files = [item for path in step.inputs for item in _files(path)]
    input_files.append(PROJECT_DIR / "scripts" / step.script)
    newest_input = max(item.stat().st_mtime for item in input_files)
    oldest_output = min(item.stat().st_mtime for files in output_files for item in files)
    return "outdated" if newest_input > oldest_output else "up to date"


def should_run(step: Step, status: str, args: argparse.Namespace) -> bool:
    selected = (
        step.group == "dashboard"
        or (step.group == "raw" and args.from_raw)
        or (step.group == "analysis" and args.all)
    )
    if status == "missing":
        return True  # nothing to fall back on, whichever group it is in
    if not selected:
        return False
    return args.force or status == "outdated"


def decide(step: Step, args: argparse.Namespace, rebuilt: set[Path]) -> tuple[str, bool]:
    """Return the step's status and whether it should run.

    `rebuilt` holds the outputs of earlier steps in this run. A step that reads
    one of them counts as outdated even if the timestamps don't show it yet.
    """
    status = step_status(step)
    if status == "up to date" and rebuilt.intersection(step.inputs):
        status = "outdated"
    return status, should_run(step, status, args)


def plan_steps(steps: list[Step], args: argparse.Namespace) -> list[tuple[Step, str, bool]]:
    """Preview of what a run would do, assuming each planned step rebuilds its outputs."""
    rebuilt: set[Path] = set()
    plan = []
    for step in steps:
        status, run = decide(step, args, rebuilt)
        if run:
            rebuilt.update(step.outputs)
        plan.append((step, status, run))
    return plan


def run_step(step: Step) -> None:
    print(f"\n==> {step.name} (scripts/{step.script})", flush=True)
    result = subprocess.run([sys.executable, str(PROJECT_DIR / "scripts" / step.script)], cwd=PROJECT_DIR)
    if result.returncode != 0:
        sys.exit(f"\nStopped: scripts/{step.script} failed (exit code {result.returncode}). Fix the error above and run again.")


def run_pipeline(steps: list[Step], args: argparse.Namespace, runner=run_step) -> list[Step]:
    """Run the steps in order and return the ones that ran.

    Each step is decided when it is reached, not up front, so a step whose
    inputs were just rebuilt by an earlier step isn't skipped on stale state.
    """
    rebuilt: set[Path] = set()
    ran = []
    for step in steps:
        _status, run = decide(step, args, rebuilt)
        if run:
            runner(step)
            ran.append(step)
            rebuilt.update(step.outputs)
        elif rebuilt.intersection(step.inputs):
            print(f"\nNote: \"{step.name}\" reads data rebuilt above but was left as committed. "
                  "Add --all to rebuild it too.")
    return ran


def launch_dashboard() -> int:
    """Run the dashboard and return Streamlit's exit code (0 when stopped with Ctrl+C)."""
    print("\nStarting the dashboard. Press Ctrl+C here to stop it.", flush=True)
    try:
        result = subprocess.run([sys.executable, "-m", "streamlit", "run", str(DASHBOARD_APP)], cwd=PROJECT_DIR)
    except KeyboardInterrupt:
        return 0
    if result.returncode != 0:
        print(f"\nThe dashboard stopped with an error (exit code {result.returncode}).", file=sys.stderr)
    return result.returncode


def check_requirements(need_dashboard: bool) -> None:
    modules = ["pandas", "numpy"] + (["streamlit", "altair"] if need_dashboard else [])
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        sys.exit(
            f"Missing Python packages: {', '.join(missing)}.\n"
            "Install everything with:  python -m pip install -r requirements.txt"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only report each step's status")
    parser.add_argument("--no-dashboard", action="store_true", help="don't start the dashboard afterwards")
    parser.add_argument("--all", action="store_true", help="also rebuild the analysis tables")
    parser.add_argument("--from-raw", action="store_true", help="also re-align the raw data (slow)")
    parser.add_argument("--force", action="store_true", help="rerun selected steps even if up to date")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not RAW_DIR.exists() and not ALIGNED_DIR.exists():
        sys.exit(f"No data found. Expected the raw data in {RAW_DIR.relative_to(PROJECT_DIR)}.")

    plan = plan_steps(STEPS, args)
    print("Pipeline steps, in order:")
    for number, (step, status, run) in enumerate(plan, start=1):
        if not run and status == "outdated" and step.group != "dashboard":
            # Committed files: after a git checkout their timestamps say little,
            # so the copy in the repo is used unless a rebuild was asked for.
            status = "in repo"
        note = "will run" if run else ""
        print(f"  {number}. {step.name:<46} {status:<11} {note}".rstrip())

    if args.check:
        if any(step.group == "dashboard" and status != "up to date" for step, status, _run in plan):
            command = "python run.py" + (" --from-raw" if args.from_raw else "")
            print(f"\nThe dashboard's results need rebuilding. Run:  {command}")
        return 0

    open_dashboard = not args.no_dashboard
    check_requirements(open_dashboard)
    ran = run_pipeline(STEPS, args)

    if not ran:
        print("\nEverything the dashboard needs is up to date.")
    if open_dashboard:
        return launch_dashboard()
    return 0


if __name__ == "__main__":
    sys.exit(main())
