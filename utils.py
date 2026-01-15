import pandas as pd

from datetime import date
from pathlib import Path 
import json
import os

import helpers

#==========
# CONTENTS 
#==========
# process_event_data
# summarize_event_data
# create_master_file
#

# Process _eventdata.csv into _eventdata_processed.csv
def process_event_data(dir: Path) -> tuple[pd.DataFrame, dict]: # single directory containing video, eventdata, metadata, etc
    # Get video name
    video_file = list(dir.glob("*.mp4"))
    if not video_file:
        video_file = list(dir.glob("*.mov"))
    video_name = video_file[0].stem

    # Get eventdata file and create df
    eventdata_filepath = dir / f"{video_name}_eventdata.csv"
    df = pd.read_csv(eventdata_filepath)

    # Read metadata 
    metadata_filepath = dir / f"{video_name}_metadata.json"
    with open(metadata_filepath, 'r', encoding="utf-8") as f:
        metadata = json.load(f)

    # Assign metadata to variables (?)
    # video_start_time, video_end_time, first_mouse_enter_time, 
    # group, cohort, date, run, mice, analyzed_date, analyzed_by

    # Variables
    square_area_cm2 = 10
    ul_per_cm2 = 25
    cutoff_time = "02:05:00"
    cutoff_time_sec = helpers.hms_to_s(cutoff_time)

    # Calibration squares
    squares_filepath = dir / f"{video_name}_squares.csv"
    squares_df = pd.read_csv(squares_filepath)
    mean_square_area_px = squares_df["Area"].mean()
    pixels_per_cm2 = mean_square_area_px / square_area_cm2

    # Clean df 
    df = df[df["Mouse"].str.lower() != "stats"] # remove "stats" rows
    
    # Helper function for new df columns
    def calculate_metrics(row):
        # Measure Delay - difference between measure time and event time
        measure_delay = helpers.subtract_hms_times(row["Actual Measure Time (s)"], row["Actual Event Time (s)"])
        
        # Event Latency - difference between event time and first mouse in time
        latency = helpers.subtract_hms_times(row["Actual Event Time (s)"], metadata["first_mouse_enter_time"])
        
        # Within Time - bool if event latency is within cutoff time
        latency_sec = helpers.hms_to_s(latency)
        within_cutoff = False 
        if pd.notna(latency) and latency_sec < cutoff_time_sec:
            within_cutoff = True

        # Area cm2 - convert void size in pixels -> cm2
        area_cm2 = row["Pixel Area"] / pixels_per_cm2

        # Volume ul raw - convert void size in cm2 to ul
        volume_ul_raw = area_cm2 * ul_per_cm2

        # Volume ul adjusted - adjust volume_ul_raw based on event latency  
        volume_ul_adjusted = helpers.adjust_volume(volume_ul_raw, latency)

        return pd.Series([measure_delay, latency, within_cutoff, area_cm2, volume_ul_raw, volume_ul_adjusted])

    # Apply helper function to df
    df[["Measure Delay", "Event Latency", "Within Time", "Area cm2", "Volume ul raw", "Volume ul adjusted"]] = df.apply(
        calculate_metrics,
        axis=1,
    )

    # Save processed df as csv
    df_save_path = dir / f"{video_name}_eventdata_processed.csv"
    df.to_csv(df_save_path, index=False)
    print(f"Saved processed eventdata -> {df_save_path}")

    return df, metadata, video_name

# Create summary for each mouse's run data
def summarize_event_data(processed_df: pd.DataFrame, metadata: dict, dir: Path, video_name: str):
    # filter out events that do not occur within cutoff time
    df_filtered = processed_df[processed_df["Within Time"]].reset_index(drop=True)

    # Loop through list of mice and get a summary for each
    summary_data = []
    for mouse in metadata["mice"]:
        mouse_data = df_filtered[df_filtered["Mouse"] == mouse]

        stats = {
            "Mouse": mouse,
            "void": (mouse_data["Type"] == "void").sum(),
            "leak": (mouse_data["Type"] == "leak").sum(),
            "Avg Void Vol (ul)": mouse_data[mouse_data["Type"] == "void"]["Volume ul adjusted"].mean(),
            "run": metadata["run"],
            "date": metadata["date"],
            "group": metadata["group"],
            "cohort": metadata["cohort"],
        }

        summary_data.append(stats)

    # Convert summary_data to df and save
    summary_df = pd.DataFrame(summary_data)
    summary_save_path = dir / f"{video_name}_summary.csv"
    summary_df.to_csv(summary_save_path, index=False)
    print(f"Saved video summary -> {summary_save_path}\n")

# Combine all _summary.csv files into MASTERDATA file
def create_master_file(parent_dir: Path, folders: list):
    master_data = []
    for folder in folders:
        summary_path = folder.with_name(folder.name.replace("_eventdata", "_summary"))
        if summary_path.exists():
            temp_df = pd.read_csv(summary_path)
            master_data.append(temp_df)
            print(f"Extracted lines from {summary_path}")
        else:
            print(f"Summary file not found for {summary_path}. Skipping...")

    if master_data:
        master_df = pd.concat(master_data, ignore_index=True)
        master_file_save_path = parent_dir / f"MASTERDATA_{date.today()}.csv"
        master_df.to_csv(master_file_save_path, index=False)
        print(f"Saved master data to {master_file_save_path}")

