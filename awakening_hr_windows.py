from pathlib import Path
import argparse
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ALIGNED_DIR = PROJECT_DIR / "aligned_data"
DEFAULT_OUTPUT = PROJECT_DIR / "awakening_hr_windows.csv"

VALID_SLEEP_STAGES = ["light", "deep", "rem"]
EXPECTED_INTERVAL = pd.Timedelta(minutes=5)


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

            current_stage = night_df["hypnogram_class"]
            previous_stage = current_stage.shift()

            time_difference = night_df[
                "datetime_utc"
            ].diff()

            # Same awakening rule used previously:
            # sleep -> awake, exactly 5 minutes apart
            awakening_mask = (
                current_stage.eq("awake")
                & previous_stage.isin(VALID_SLEEP_STAGES)
                & time_difference.eq(EXPECTED_INTERVAL)
            )

            awakening_rows = night_df[awakening_mask]

            for _, awakening in awakening_rows.iterrows():

                awakening_time = awakening["datetime_utc"]

                result = {
                    "participant_id": participant_id,
                    "date": night_date.strftime("%Y-%m-%d"),
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

    awakening_windows.to_csv(
        args.output,
        index=False
    )

    print(
        f"\nFound {len(awakening_windows)} awakenings."
    )

    print(
        f"Saved HR windows to: {args.output}"
    )

    print("\nSample:")
    print(awakening_windows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()