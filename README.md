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
   Load sleep-stage, heart-rate, skin-temperature, and timestamp data.
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

```mermaid
graph TD
    A[Longitudinal Data<br/>320 nights] --> B[Data Loading & Preprocessing]
    
    subgraph Files
        B --> B1[sleep_hypnogram.csv<br/>5-min Sleep Stages]
        B --> B2[heart_rate.csv<br/>5-min HR / HRV]
        B --> B3[sleep.csv<br/>Temp Delta & Oura validation]
    end

    Files --> C[Nightly Feature Extraction]
    
    C -->|Sleep| C1[Total, Deep, REM, Light, Awake, Stage Changes]
    C -->|Heart| C2[Avg HR, Min HR, HRV, HR by Stage]
    C -->|Temp| C3[Nightly Temp Delta]

    C1 & C2 & C3 --> D[Create Nightly Dataset]
    D --> E[Rolling Personal Baseline<br/>Previous 30 Nights]
    E --> F[Selected Night]
    F --> G[Compare Selected Night vs Baseline]
    G --> H[Change / Anomaly Detection]

    H --> I[Change Detection<br/>Unusual deviation]
    H --> J[Pattern Analysis<br/>Cross-night correlations]
    H --> K[Normal Night<br/>No meaningful deviation]

    I & J & K --> L[Dashboard + LLM Insight]
