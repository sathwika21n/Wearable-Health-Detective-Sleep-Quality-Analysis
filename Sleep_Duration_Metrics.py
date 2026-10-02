import os
import glob
import pandas as pd

def generate_participant_reports(aligned_dir="./aligned_data", output_dir="./participant_excel_reports"):
    os.makedirs(output_dir, exist_ok=True)
    file_pattern = os.path.join(aligned_dir, "par_*_aligned.csv")
    csv_files = glob.glob(file_pattern)

    if not csv_files:
        print(f"No files matching '{file_pattern}' found.")
        return

    print(f"Found {len(csv_files)} participant file(s) to process.\n")

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        participant_id = filename.split('_aligned')[0]

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[{participant_id}] Error reading {filepath}: {e}")
            continue

        if 'date' not in df.columns or 'hypnogram_class' not in df.columns:
            print(f"[{participant_id}] Missing required columns 'date' or 'hypnogram_class'.")
            continue

        nightly_rows = []

        # Calculate nightly duration
        for sleep_date, group in df.groupby('date'):
            if pd.isna(sleep_date):
                continue

            clean_stages = group['hypnogram_class'].astype(str).str.strip().str.lower()
            
            # Count sleep epochs (excluding awake & unassigned)
            sleep_epochs = clean_stages.isin(['light', 'deep', 'rem', 'sleep', 'n1', 'n2', 'n3']).sum()
            calc_tst_min = round(sleep_epochs * 5.0, 2)  # 5 min per epoch

            # Check for unassigned/missing epochs
            unassigned_epochs = (~clean_stages.isin(['light', 'deep', 'rem', 'sleep', 'n1', 'n2', 'n3', 'awake'])).sum()
            gap_note = "None"
            if unassigned_epochs > 0:
                gap_note = f"{unassigned_epochs} missing/unassigned epoch(s) ({unassigned_epochs * 5} min omitted)"

            # Source summary sleep time (convert seconds to minutes)
            if 'night_total_sleep_seconds' in group.columns and not group['night_total_sleep_seconds'].dropna().empty:
                source_tst_min = round(group['night_total_sleep_seconds'].dropna().iloc[0] / 60.0, 2)
            else:
                source_tst_min = None

            nightly_rows.append({
                'Participant ID': participant_id,
                'Date': sleep_date,
                'Calculated TST (min)': calc_tst_min,
                'Source Summary TST (min)': source_tst_min if source_tst_min is not None else 'N/A',
                'Gap / Missing Data Notes': gap_note
            })

        if not nightly_rows:
            continue

        p_df = pd.DataFrame(nightly_rows)
        p_df['Date_dt'] = pd.to_datetime(p_df['Date'])

        # Numeric conversion for average calculations
        p_df['Source_Numeric'] = pd.to_numeric(p_df['Source Summary TST (min)'], errors='coerce')

        # 1. Weekly Averages
        p_df['Week'] = p_df['Date_dt'].dt.to_period('W').astype(str)
        weekly_avg = p_df.groupby('Week', as_index=False)[['Calculated TST (min)', 'Source_Numeric']].mean().round(2)
        weekly_avg.rename(columns={'Week': 'Timeframe', 'Source_Numeric': 'Source Summary TST (min)'}, inplace=True)
        weekly_avg['Timeframe'] = 'Week ' + weekly_avg['Timeframe']

        # 2. Monthly Averages
        p_df['Month'] = p_df['Date_dt'].dt.to_period('M').astype(str)
        monthly_avg = p_df.groupby('Month', as_index=False)[['Calculated TST (min)', 'Source_Numeric']].mean().round(2)
        monthly_avg.rename(columns={'Month': 'Timeframe', 'Source_Numeric': 'Source Summary TST (min)'}, inplace=True)
        monthly_avg['Timeframe'] = 'Month ' + monthly_avg['Timeframe']

        # 3. Overall Average
        overall_avg = pd.DataFrame([{
            'Metric': 'OVERALL AVERAGE',
            'Calculated TST (min)': round(p_df['Calculated TST (min)'].mean(), 2),
            'Source Summary TST (min)': round(p_df['Source_Numeric'].mean(), 2)
        }])

        # PURE Nightly Table (Page 1)
        nightly_export = p_df[['Participant ID', 'Date', 'Calculated TST (min)', 'Source Summary TST (min)', 'Gap / Missing Data Notes']].copy()

        # Save to Multi-Tab Excel Output
        output_file = os.path.join(output_dir, f"{participant_id}_sleep_summary.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            nightly_export.to_excel(writer, sheet_name='Nightly Sleep Summary', index=False)
            weekly_avg.to_excel(writer, sheet_name='Weekly Averages', index=False)
            monthly_avg.to_excel(writer, sheet_name='Monthly Averages', index=False)
            overall_avg.to_excel(writer, sheet_name='Overall Summary', index=False)

        print(f"Generated Excel report for {participant_id} -> {output_file}")

    print("\nProcessing complete for all participants!")

if __name__ == "__main__":
    generate_participant_reports()