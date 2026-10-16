import os
import glob
import pandas as pd
import numpy as np

def extract_participant_num(filename):
    """Extracts integer participant number for natural sorting (par_1 -> 1, par_24 -> 24)."""
    basename = os.path.basename(filename)
    pid_str = basename.split('_aligned')[0].replace('par_', '')
    try:
        return int(pid_str)
    except ValueError:
        return 999

def process_participant_heart_rate(aligned_dir=".", output_file="participant_heart_rate_summary.xlsx"):
    """
    Summarizes row-level heart rate readings across sleep windows and sleep stages.
    
    Tab 1: Nightly Sleeping HR Summary
    Tab 2: HR-by-Stage Summary (per night)
    Tab 3: Participant Stage Averages (WEIGHTED by sample count per night)
    """
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    file_pattern = os.path.join(aligned_dir, "par_*_aligned*.csv")
    csv_files = glob.glob(file_pattern)
    
    if not csv_files:
        csv_files = glob.glob("**/par_*_aligned*.csv", recursive=True)

    if not csv_files:
        print(f"No participant aligned CSV files found in directory: '{aligned_dir}'")
        return

    csv_files = sorted(set(csv_files), key=extract_participant_num)
    print(f"Found {len(csv_files)} participant file(s) to process in order from par_1 to par_24.\n")

    nightly_hr_rows = []
    stage_hr_rows = []

    STAGE_MAP = {
        'light': 'Light', 'n1': 'Light', 'n2': 'Light', 'sleep-n1': 'Light', 'sleep-n2': 'Light',
        'deep': 'Deep', 'n3': 'Deep', 'sleep-n3': 'Deep',
        'rem': 'REM', 'sleep-rem': 'REM',
        'awake': 'Awake', 'wake': 'Awake'
    }

    SLEEPING_STAGES = {'Light', 'Deep', 'REM'}

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        participant_id = filename.split('_aligned')[0]

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"[{participant_id}] Error reading {filepath}: {e}")
            continue

        required_cols = ['date', 'heart_rate', 'hypnogram_class']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"[{participant_id}] Missing required columns: {missing_cols}. Skipping.")
            continue

        df['date_str'] = df['date'].astype(str)
        sorted_dates = sorted(df['date_str'].dropna().unique())

        for sleep_date in sorted_dates:
            group = df[df['date_str'] == sleep_date]

            # Filter valid row-level heart rate readings (> 0 bpm and non-null)
            valid_hr_df = group[group['heart_rate'].notna() & (group['heart_rate'] > 0)].copy()
            
            valid_hr_df['clean_stage'] = valid_hr_df['hypnogram_class'].astype(str).str.strip().str.lower()
            valid_hr_df['mapped_stage'] = valid_hr_df['clean_stage'].map(STAGE_MAP)

            # 1. Nightly Sleeping HR Summary (Excluding Awake)
            sleeping_mask = valid_hr_df['mapped_stage'].isin(SLEEPING_STAGES)
            sleeping_hr = valid_hr_df.loc[sleeping_mask, 'heart_rate']
            valid_samples_count = len(sleeping_hr)

            if valid_samples_count > 0:
                mean_hr = round(sleeping_hr.mean(), 2)
                median_hr = round(sleeping_hr.median(), 2)
                min_hr = round(sleeping_hr.min(), 2)
                max_hr = round(sleeping_hr.max(), 2)
            else:
                mean_hr = median_hr = min_hr = max_hr = "N/A"

            nightly_hr_rows.append({
                'Participant ID': participant_id,
                'Sleep Window (Date)': str(sleep_date),
                'Mean HR (bpm)': mean_hr,
                'Median HR (bpm)': median_hr,
                'Min HR (bpm)': min_hr,
                'Max HR (bpm)': max_hr,
                'Valid Sample Count': valid_samples_count
            })

            # 2. Stage HR Summary (Light, Deep, REM, Awake)
            for stage in ['Light', 'Deep', 'REM', 'Awake']:
                stage_hr = valid_hr_df.loc[valid_hr_df['mapped_stage'] == stage, 'heart_rate']
                stage_samples_count = len(stage_hr)

                if stage_samples_count > 0:
                    s_mean = round(stage_hr.mean(), 2)
                    s_median = round(stage_hr.median(), 2)
                    s_sum = stage_hr.sum()  # Total HR sum for weighted average pooling
                else:
                    s_mean = s_median = "N/A"
                    s_sum = 0.0

                stage_hr_rows.append({
                    'Participant ID': participant_id,
                    'Sleep Window (Date)': str(sleep_date),
                    'Stage': stage,
                    'Mean HR (bpm)': s_mean,
                    'Median HR (bpm)': s_median,
                    'Sample Count': stage_samples_count,
                    'Stage_HR_Sum': s_sum
                })

    if not nightly_hr_rows:
        print("No valid heart rate records processed.")
        return

    nightly_hr_df = pd.DataFrame(nightly_hr_rows)
    stage_hr_df = pd.DataFrame(stage_hr_rows)

    # 3. Weighted Participant Stage Averages (Tab 3)
    participant_stage_rows = []
    
    stage_hr_df['p_num'] = stage_hr_df['Participant ID'].apply(extract_participant_num)
    sorted_pids = stage_hr_df.sort_values('p_num')['Participant ID'].unique()

    for pid in sorted_pids:
        p_group = stage_hr_df[stage_hr_df['Participant ID'] == pid]
        row = {'Participant ID': pid}
        
        for stage in ['Light', 'Deep', 'REM', 'Awake']:
            stage_group = p_group[p_group['Stage'] == stage]
            total_samples = stage_group['Sample Count'].sum()
            total_hr_sum = stage_group['Stage_HR_Sum'].sum()
            
            if total_samples > 0:
                # Weighted Mean = Total Sum of Readings / Total Valid Samples
                weighted_mean = round(total_hr_sum / total_samples, 2)
                
                # Weighted Median Approximation / Average Nightly Median
                valid_medians = pd.to_numeric(stage_group['Median HR (bpm)'], errors='coerce')
                weights = stage_group['Sample Count']
                valid_mask = valid_medians.notna() & (weights > 0)
                
                if valid_mask.any():
                    weighted_median = round(np.average(valid_medians[valid_mask], weights=weights[valid_mask]), 2)
                else:
                    weighted_median = "N/A"

                row[f'{stage} Weighted Mean HR (bpm)'] = weighted_mean
                row[f'{stage} Weighted Median HR (bpm)'] = weighted_median
                row[f'{stage} Total Valid Samples'] = total_samples
            else:
                row[f'{stage} Weighted Mean HR (bpm)'] = "N/A"
                row[f'{stage} Weighted Median HR (bpm)'] = "N/A"
                row[f'{stage} Total Valid Samples'] = 0
                
        participant_stage_rows.append(row)

    participant_stage_df = pd.DataFrame(participant_stage_rows)

    # Clean internal calculation columns before export
    if 'Stage_HR_Sum' in stage_hr_df.columns:
        stage_hr_df.drop(columns=['Stage_HR_Sum', 'p_num'], inplace=True)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        nightly_hr_df.to_excel(writer, sheet_name='Nightly Sleeping HR', index=False)
        stage_hr_df.to_excel(writer, sheet_name='HR by Stage', index=False)
        participant_stage_df.to_excel(writer, sheet_name='Participant Stage Averages', index=False)

    print(f"Successfully generated weighted HR summary workbook -> '{output_file}'")

if __name__ == "__main__":
    process_participant_heart_rate()