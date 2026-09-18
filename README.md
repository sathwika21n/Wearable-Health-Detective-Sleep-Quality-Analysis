# Wearable-Health-Detective-Sleep-Quality-Analysis

Team 7
Team members: Sathwika Thatiparthi, Hai Nguyen, Nitin Thota, Abhiram Gunna, Youngjun Ryoo, Joshua Feng

# Importance of the Topic

Sleep is an important health metric because poor or irregular sleep can affect a person's overall health, daily performance, and well-being. Wearable devices can collect large amounts of sleep-related data, but users may have difficulty understanding what the data means or recognizing changes in their sleep patterns.

Our project aims to analyze wearable sleep data to identify important patterns such as sleep duration, sleep stages, awakenings, and changes in sleep quality. By combining sleep information with other measurements such as heart rate and skin temperature, the system can identify relationships between different health metrics.

The goal is not just to display sleep data, but to act as a "Sleep Detective" that helps users understand what changed, when it changed, and possible factors associated with the change. This directly addresses the data-overload problem discussed in the course slides, where wearable devices generate thousands of data points but users may not understand why their health metrics changed.
The project will primarily analyze three modalities:
Sleep
Heart Rate
Skin Temperature

# Project Goals

Our project has three main goals:
Analyze sleep quality and sleep patterns
Use the existing sleep-stage labels in the DREAMT dataset.
Calculate metrics such as total sleep time, time spent in each sleep stage, and number of awakenings.
Analyze relationships between sleep, heart rate, and skin temperature
Compare sleep-related metrics with heart rate and skin temperature.
Identify correlations and patterns without claiming direct causation.
Present sleep trends and insights to the user
Display results through graphs, timelines, and dashboards.
Provide simple, user-friendly explanations of important sleep changes.

# Pipeline

1. Data Input
   Load DREAMT sleep-stage, heart-rate, skin-temperature, and timestamp data.
2. Data Preprocessing
   Clean and organize the dataset.
   Handle missing values.
   Align the different signals by time.
3. Sleep Analysis
   Calculate:
   Total sleep time
   Time in each sleep stage
   Deep sleep duration
   REM sleep duration
   Wake periods
   Number of awakenings
4. Heart Rate Analysis
   Calculate sleeping heart-rate statistics.
   Analyze heart-rate changes during different sleep stages.
5. Skin Temperature Analysis
   Analyze nighttime skin-temperature trends.
   Compare temperature across different sleep stages.
6. Multi-Modal Analysis
   Compare sleep metrics with heart-rate and skin-temperature measurements.
   Identify statistical relationships and useful patterns.
7. Pattern / Change Detection
   Detect unusual or meaningful sleep changes.
   Use statistical and rule-based methods to identify changes from normal patterns.
8. Visualization
   Display results using graphs, timelines, and dashboards.
9. User Insights
   Provide simple explanations of detected sleep patterns and changes.
   Optional AI-generated summaries may be added if time allows.

# Project Flow

Longitudinal Data
(320 nights for one participant)

        ↓

Data Loading & Preprocessing

sleep_hypnogram.csv
→ 5-min Sleep Stages

heart_rate.csv
→ 5-min HR / HRV

sleep.csv
→ Temperature Delta
→ Oura summary for validation

        ↓

Nightly Feature Extraction

Sleep:

- Total Sleep Time
- Deep Sleep
- REM Sleep
- Light Sleep
- Awake Time
- Sleep Stage Changes

Heart:

- Average HR
- Minimum HR
- HRV
- HR by Sleep Stage

Temperature:

- Nightly Temperature Delta

        ↓

Create Nightly Dataset

## Date | Sleep | Deep | REM | Awake | Avg HR | HRV | Temp

Day 1
Day 2
Day 3
...
Day 320

        ↓

Rolling Personal Baseline
(Previous 30 Nights)

- Mean / Median
- Standard Deviation
- Normal Range

        ↓

Selected Night

        ↓

Compare Selected Night
with Previous 30-Night Baseline

        ↓

Change / Anomaly Detection

Examples:

Sleep Duration ↓
Deep Sleep ↓
Heart Rate ↑
HRV ↓
Temperature Delta ↑

        ↓

┌────────────────────┬─────────────────────┐
│ │ │
Change Detection Pattern Analysis Normal Night
│ │ │
↓ ↓ ↓
Unusual deviation Relationships No meaningful
from baseline across many nights deviation

                     Examples:
                     Sleep ↓ ↔ HR ↑
                     Deep Sleep ↓ ↔ HRV ↓
                     Sleep ↓ ↔ Temp change

        ↓

Dashboard + LLM Insight
