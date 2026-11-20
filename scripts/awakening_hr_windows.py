from datetime import datetime
from pathlib import Path
import argparse
import pandas as pd

from awakening_detection import normalize_stages, sleep_to_awake_mask


PROJECT_DIR = Path(__file__).resolve().parent.parent
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "awakening_hr_windows_rem.csv"
STAGE_DISPLAY_NAMES = {"light": "Light", "deep": "Deep", "rem": "REM"}


def write_output(data, output_path):
    """Write the CSV, falling back if the requested file is locked."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(output_path, index=False)
        return output_path
    except PermissionError:
        fallback_stem = output_path.stem
        if not fallback_stem.endswith("_rem"):
            fallback_stem = f"{fallback_stem}_rem"
        fallback_path = output_path.with_name(
            f"{fallback_stem}{output_path.suffix}"
        )
        try:
            data.to_csv(fallback_path, index=False)
        except PermissionError:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fallback_path = output_path.with_name(
                f"{fallback_stem}_{timestamp}{output_path.suffix}"
            )
            data.to_csv(fallback_path, index=False)
        print(
            f"Could not overwrite {output_path}; it may be open in another "
            f"program. Wrote the output to {fallback_path} instead."
        )
        return fallback_path


def extract_awakening_hr_windows(
    aligned_dir,
    window_minutes=30
):
    rows = []

    csv_files = sorted(Path(aligned_dir).glob("par_*_aligned.csv"))

    for csv_path in csv_files:
        participant_id = csv_path.stem.replace("_aligned", "")

        print(f"Processing {participant_id}...")

        df = pd.read_csv(csv_path)

        df["datetime_utc"] = pd.to_datetime(
            df["datetime_utc"],
            utc=True,
            errors="coerce"
        )

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values("datetime_utc").reset_index(drop=True)

        # Process each night separately
        for night_date, night_df in df.groupby("date"):
            night_df = night_df.sort_values(
                "datetime_utc"
            ).copy()

            # Same awakening rule used previously:
            # sleep -> awake, exactly 5 minutes apart
            awakening_mask = sleep_to_awake_mask(
                night_df["hypnogram_class"],
                night_df["datetime_utc"],
            )

            previous_stages = normalize_stages(
                night_df["hypnogram_class"]
            ).shift()
            awakening_rows = night_df[awakening_mask]

            for row_index, awakening in awakening_rows.iterrows():

                awakening_time = awakening["datetime_utc"]
                previous_stage = previous_stages.loc[row_index]

                result = {
                    "participant_id": participant_id,
                    "date": night_date.strftime("%Y-%m-%d"),
                    "previous_sleep_stage": STAGE_DISPLAY_NAMES[previous_stage],
                    "awakening_time_utc": awakening_time,
                }

                # Example for 30 min:
                # -30, -25, -20, -15, -10, -5
                offsets = range(
                    window_minutes,
                    0,
                    -5
                )

                for minutes_before in offsets:

                    expected_time = (
                        awakening_time
                        - pd.Timedelta(minutes=minutes_before)
                    )

                    hr_match = night_df[
                        night_df["datetime_utc"].eq(
                            expected_time
                        )
                    ]

                    column_name = f"hr_minus_{minutes_before}m"

                    if (
                        not hr_match.empty
                        and hr_match["heart_rate"].notna().any()
                    ):
                        result[column_name] = (
                            hr_match["heart_rate"]
                            .dropna()
                            .iloc[0]
                        )
                    else:
                        # Do NOT interpolate missing HR samples
                        result[column_name] = pd.NA

                rows.append(result)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract heart-rate samples before each "
            "sleep-to-awake transition."
        )
    )

    parser.add_argument(
        "--window-minutes",
        type=int,
        default=30,
        help="Minutes of HR history before each awakening (default: 30)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    awakening_windows = extract_awakening_hr_windows(
        ALIGNED_DIR,
        window_minutes=args.window_minutes
    )

    awakening_windows = awakening_windows.sort_values(
        [
            "participant_id",
            "date",
            "awakening_time_utc"
        ]
    )

    output_path = write_output(awakening_windows, args.output)

    print(
        f"\nFound {len(awakening_windows)} awakenings."
    )

    print(
        f"Saved HR windows to: {output_path}"
    )

    print("\nSample:")
    print(awakening_windows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
