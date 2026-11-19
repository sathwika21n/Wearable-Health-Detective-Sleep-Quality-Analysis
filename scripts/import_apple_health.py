"""Convert Apple Health export.xml into local Sleep Detective CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean


PARTICIPANT = "par_apple"
EPOCH = timedelta(minutes=5)
SESSION_GAP = timedelta(hours=2)
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
APPLE_STAGES = {
    "HKCategoryValueSleepAnalysisInBed": "in_bed",
    "HKCategoryValueSleepAnalysisAsleep": "asleep",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "asleep",
    "HKCategoryValueSleepAnalysisAsleepCore": "light",
    "HKCategoryValueSleepAnalysisAsleepDeep": "deep",
    "HKCategoryValueSleepAnalysisAsleepREM": "REM",
    "HKCategoryValueSleepAnalysisAwake": "awake",
}
LEVELS = {"deep": 1, "light": 2, "REM": 3, "awake": 4}
ASLEEP = {"deep", "light", "REM", "asleep"}
HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
SDNN = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
TEMPERATURE = "HKQuantityTypeIdentifierAppleSleepingWristTemperature"
ALIGNED_COLUMNS = [
    "date", "datetime_utc", "sleep_start_utc", "sleep_end_utc",
    "heart_rate", "heart_rmssd", "hypnogram_level", "hypnogram_class",
    "temperature_delta", "night_time_in_bed_seconds", "night_total_sleep_seconds",
    "night_deep_sleep_seconds", "night_rem_sleep_seconds", "night_light_sleep_seconds",
    "night_awake_seconds", "night_sleep_efficiency_percent", "night_hr_average",
    "night_hr_lowest", "night_hrv_rmssd", "night_observed_awakenings",
    "night_observed_stage_changes", "heart_sdnn_ms", "wrist_temperature_c",
]
AWAKENING_COLUMNS = ["participant_id", "date", "awakening_count", "total_awake_minutes"]


@dataclass(frozen=True)
class SleepInterval:
    start: datetime
    end: datetime
    stage: str
    wake_date: str


@dataclass(frozen=True)
class Sample:
    start: datetime
    end: datetime
    value: float


SleepSegment = tuple[datetime, datetime, str]


def parse_date(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("Apple Health timestamps must include a UTC offset")
    return result


def normalize_quantity(kind: str, value: float, unit: str) -> float | None:
    if not math.isfinite(value):
        return None
    if kind == HEART_RATE and unit in {"count/min", "bpm"}:
        return value if value > 0 else None
    if kind == SDNN and unit in {"ms", "s"}:
        return value * (1000 if unit == "s" else 1) if value >= 0 else None
    if kind == TEMPERATURE:
        if unit in {"degC", "°C"}:
            return value
        if unit in {"degF", "°F"}:
            return (value - 32) * 5 / 9
    return None


def read_export(path: Path) -> tuple[list[SleepInterval], dict[str, list[Sample]]]:
    """Read only supported records and release processed XML elements."""
    sleep: set[SleepInterval] = set()
    samples: dict[str, set[Sample]] = {kind: set() for kind in (HEART_RATE, SDNN, TEMPERATURE)}
    events = ET.iterparse(path, events=("start", "end"))
    _, root = next(events)
    depth = 0
    for event, element in events:
        if event == "start":
            depth += 1
            continue
        if depth == 1:
            if element.tag == "Record":
                record = element.attrib
                kind = record.get("type")
                stage = APPLE_STAGES.get(record.get("value", "")) if kind == SLEEP_TYPE else None
                if stage or kind in samples:
                    try:
                        start = parse_date(record["startDate"]).astimezone(timezone.utc)
                        local_end = parse_date(record["endDate"])
                        end = local_end.astimezone(timezone.utc)
                        if stage and end > start:
                            sleep.add(SleepInterval(start, end, stage, local_end.date().isoformat()))
                        elif kind in samples and end >= start:
                            value = normalize_quantity(kind, float(record["value"]), record.get("unit", ""))
                            if value is not None:
                                samples[kind].add(Sample(start, end, value))
                    except (KeyError, ValueError, OverflowError):
                        # Malformed or unsupported records have no usable measurement.
                        pass
            root.remove(element)
        depth -= 1
    return sorted(sleep, key=lambda record: (record.start, record.end, record.stage)), {
        kind: sorted(records, key=lambda sample: (sample.start, sample.end, sample.value))
        for kind, records in samples.items()
    }


def covered_segments(intervals: list[SleepInterval]) -> list[SleepSegment]:
    """Resolve overlaps without counting the same seconds twice."""
    boundaries = sorted({point for interval in intervals for point in (interval.start, interval.end)})
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        stages = {item.stage for item in intervals if item.start < end and item.end > start}
        detailed = stages & LEVELS.keys()
        if len(detailed) == 1:
            stage = next(iter(detailed))
        elif not detailed and "asleep" in stages:
            stage = "asleep"
        else:
            stage = "unknown"
        segments.append((start, end, stage))
    return segments


def sleep_sessions(intervals: list[SleepInterval]) -> dict[str, list[SleepInterval]]:
    """Choose the session with the most observed sleep for each local wake date."""
    groups: list[list[SleepInterval]] = []
    group_end = None
    for interval in intervals:
        if interval.stage == "in_bed":
            continue
        if group_end is None or interval.start - group_end >= SESSION_GAP:
            groups.append([])
            group_end = interval.end
        groups[-1].append(interval)
        group_end = max(group_end, interval.end)

    sessions = {}
    scores = {}
    for group in groups:
        asleep = [item for item in group if item.stage in ASLEEP]
        if not asleep:
            continue
        wake_date = max(asleep, key=lambda item: (item.end, item.wake_date)).wake_date
        score = sum((end - start).total_seconds() for start, end, stage in covered_segments(group)
                    if stage in ASLEEP)
        if score > scores.get(wake_date, -1):
            scores[wake_date] = score
            sessions[wake_date] = group
    return sessions


def epoch_stage(segments: list[SleepSegment], start: datetime, end: datetime) -> str | None:
    seconds = defaultdict(float)
    seconds["unknown"] = (end - start).total_seconds()
    for left, right, stage in segments:
        if stage in LEVELS or stage == "asleep":
            overlap = (min(end, right) - max(start, left)).total_seconds()
            if overlap > 0:
                seconds[stage] += overlap
                seconds["unknown"] -= overlap
    most = max(seconds.values())
    winners = [stage for stage, duration in seconds.items() if duration == most]
    return winners[0] if len(winners) == 1 and winners[0] in LEVELS else None


def average(values: list[float]) -> float | None:
    return mean(values) if values else None


def utc_text(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def align_night(
    date: str,
    intervals: list[SleepInterval],
    in_bed: list[SleepInterval],
    samples: dict[str, list[Sample]],
) -> tuple[list[dict], dict]:
    start = intervals[0].start
    end = max(item.end for item in intervals)
    segments = covered_segments(intervals)
    durations = defaultdict(float)
    for left, right, stage in segments:
        durations[stage] += (right - left).total_seconds()
    total_sleep = sum(durations[stage] for stage in ASLEEP)
    bed = next((item for item in in_bed if item.start <= start and item.end >= end), None)
    in_bed_seconds = (bed.end - bed.start).total_seconds() if bed else None
    if bed:
        start, end = bed.start, bed.end

    bins = {kind: defaultdict(list) for kind in (HEART_RATE, SDNN)}
    sleeping_hr = []
    for kind in bins:
        records = samples[kind]
        first = bisect_left(records, start, key=lambda sample: sample.start)
        last = bisect_left(records, end, key=lambda sample: sample.start)
        for sample in records[first:last]:
            bins[kind][(sample.start - start) // EPOCH].append(sample.value)
            if kind == HEART_RATE and any(
                left <= sample.start < right and stage in ASLEEP for left, right, stage in segments
            ):
                sleeping_hr.append(sample.value)
    # An interval can span the whole sleep window. A point reading at the
    # beginning of the window is also eligible.
    temperatures = [
        sample.value for sample in samples[TEMPERATURE]
        if sample.start < end and (sample.end > start or sample.start >= start)
    ]
    rows = []
    epoch = start
    while epoch < end:
        stage = epoch_stage(segments, epoch, min(epoch + EPOCH, end))
        index = len(rows)
        rows.append({
            "date": date, "datetime_utc": utc_text(epoch),
            "sleep_start_utc": utc_text(start), "sleep_end_utc": utc_text(end),
            "heart_rate": average(bins[HEART_RATE][index]),
            "heart_sdnn_ms": average(bins[SDNN][index]),
            "hypnogram_class": stage, "hypnogram_level": LEVELS.get(stage),
        })
        epoch += EPOCH

    stages = [row["hypnogram_class"] for row in rows]
    transitions = [(a, b) for a, b in zip(stages, stages[1:]) if a and b and a != b]
    awakenings = sum(b == "awake" for _, b in transitions) if any(stages) else None
    nightly = {
        "night_time_in_bed_seconds": in_bed_seconds,
        "night_sleep_efficiency_percent": 100 * total_sleep / in_bed_seconds if in_bed_seconds else None,
        "night_total_sleep_seconds": total_sleep or None,
        "night_deep_sleep_seconds": durations["deep"] or None,
        "night_light_sleep_seconds": durations["light"] or None,
        "night_rem_sleep_seconds": durations["REM"] or None,
        "night_awake_seconds": durations["awake"] or None,
        "night_hr_average": average(sleeping_hr),
        "night_hr_lowest": min(sleeping_hr, default=None),
        "night_observed_awakenings": awakenings,
        "night_observed_stage_changes": len(transitions) if any(stages) else None,
        "wrist_temperature_c": average(temperatures),
    }
    for row in rows:
        row.update(nightly)
    return rows, {
        "participant_id": PARTICIPANT, "date": date, "awakening_count": awakenings,
        "total_awake_minutes": stages.count("awake") * 5 if any(stages) else None,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def import_export(xml_path: Path, output_dir: Path) -> tuple[Path, Path]:
    intervals, samples = read_export(xml_path)
    in_bed = sorted(
        (item for item in intervals if item.stage == "in_bed"),
        key=lambda item: (item.end - item.start, item.start),
    )
    sessions = sleep_sessions(intervals)
    if not sessions:
        raise ValueError("No usable sleep sessions found in the Apple Health export")

    aligned, awakenings = [], []
    for date, session in sorted(sessions.items()):
        rows, awakening = align_night(date, session, in_bed, samples)
        aligned.extend(rows)
        awakenings.append(awakening)
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = output_dir / f"{PARTICIPANT}_aligned.csv"
    awakening_path = output_dir / "nightly_awakenings.csv"
    write_csv(aligned_path, ALIGNED_COLUMNS, aligned)
    write_csv(awakening_path, AWAKENING_COLUMNS, awakenings)
    return aligned_path, awakening_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_xml", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="Dedicated local output folder")
    args = parser.parse_args()
    try:
        aligned, awakenings = import_export(args.export_xml, args.output_dir)
    except (OSError, ValueError, ET.ParseError) as exc:
        parser.exit(1, f"Apple Health import failed: {exc}\n")
    print(f"Wrote {aligned} and {awakenings}")


if __name__ == "__main__":
    main()
