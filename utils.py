import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
# create_group_summary_bymouse
# create_group_summary_byrun
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
        print(f"\nSaved master data to {master_file_save_path}")

def create_group_summary_bymouse(group_folder: Path):
    # Get all summary files inside group folder
    summary_files = list(group_folder.rglob("*_summary.csv"))

    # Collect data from all summary files
    group_data = []
    for file in summary_files:
        temp_df = pd.read_csv(file)
        group_data.append(temp_df)
    df = pd.concat(group_data, ignore_index=True)

    # Sort and reorder df
    df = df.sort_values(by=["Mouse", "run"])
    df = df[["cohort", "Mouse", "run", "void", "leak", "Avg Void Vol (ul)", "date", "group"]]

    # Set up logic for creating summary file 
    mouse_data = list(df.groupby("Mouse"))
    total_mice = len(mouse_data)

    results = []
    averages = []
    empty_row_df = pd.DataFrame([[None] * len(df.columns)], columns=df.columns).fillna("")

    # Loop through mouse_data and add summarized version to results and averages
    for i, (mouse, group) in enumerate(mouse_data):
        # Add Run1, Run2 data to results
        results.append(group)

        # Calculate summary values
        avg_void = group['void'].mean()
        avg_leak = group['leak'].mean()
        wavv = (group["Avg Void Vol (ul)"] * group["void"]).sum() / group["void"].sum() if group["void"].sum() > 0 else np.nan

        # Create summary row
        summary_row = group.iloc[[0]].copy()
        summary_row["void"] = avg_void
        summary_row["leak"] = avg_leak
        summary_row["Avg Void Vol (ul)"] = wavv
        summary_row['run'] = "AVERAGE"

        # Add summary row to results then an empty row
        results.append(summary_row)
        results.append(empty_row_df)

        # Add summary row to averages. Add three empty rows once all summaries have been completed
        averages.append(summary_row)
        if (i == total_mice - 1): 
            averages.append(empty_row_df)
            averages.append(empty_row_df)
            averages.append(empty_row_df)
        
    # Combine averages and results lists so that the summary csv has averages on the top, then followed by data for each mouse
    summary_start_with_avg = averages + results
    summary_df = pd.concat(summary_start_with_avg, ignore_index=True)

    # Save csv
    group_name = group_folder.stem
    summary_csv_path = group_folder / f"{group_name.upper()}_SUMMARY_BY_MOUSE.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary by mouse saved to: {summary_csv_path}")

def create_group_summary_byrun(group_folder: Path):
    # Get all summary files inside group folder
    summary_files = list(group_folder.rglob("*_summary.csv"))

    # Collect data from all summary files
    group_data = []
    for file in summary_files:
        temp_df = pd.read_csv(file)
        group_data.append(temp_df)
    df = pd.concat(group_data, ignore_index=True)

    # Sort and reorder df
    df = df.sort_values(by=["Mouse", "run"])
    df = df[["cohort", "run", "Mouse", "void", "leak", "Avg Void Vol (ul)", "date", "group"]]

    # Set up logic for creating summary file
    results = []

    # Loop through df grouped by run and add summarized version to results
    for run, mouse_data in df.groupby("run"):
        # Add mouse data by run to results list
        results.append(mouse_data)

        # Calculate summary values
        avg_void = mouse_data['void'].mean()
        avg_leak = mouse_data['leak'].mean()
        avv = mouse_data["Avg Void Vol (ul)"].mean()

        # Create summary row
        summary_row = mouse_data.iloc[0].copy()
        summary_row["void"] = avg_void
        summary_row["leak"] = avg_leak
        summary_row["Avg Void Vol (ul)"] = avv
        summary_row['Mouse'] = "AVERAGE"

        # Empty row
        empty_row = {col: [""] for col in df.columns}

        # Add summary for the run followed by an empty row
        results.append(pd.DataFrame([summary_row]))
        results.append(pd.DataFrame(empty_row))

    # Create run summary df and save as csv
    summary_df = pd.concat(results, ignore_index=True)
    group_name = group_folder.stem
    summary_csv_path = group_folder / f"{group_name.upper()}_SUMMARY_BY_RUN.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary by run saved to: {summary_csv_path}\n")




####################################################
####################################################
# # PLOT
####################################################
# plot_sequential_data
####################################################

def plot_sequential_data(root_dir: Path, plot_title:str, groups_to_plot: list, mice_to_exclude: list, runs_to_include: list):
    # Set up directories
    today = date.today().strftime("%Y%m%d")
    save_path = root_dir / "__PLOTS__" / f"{plot_title}_{today}"
    save_path.mkdir(parents = True, exist_ok = True)

    # Load MASTERDATA csv as master df
    master_csv_path = list(root_dir.glob("MASTERDATA*"))[0]
    master_df = pd.read_csv(master_csv_path)

    # Filter and convert run column to str
    processed_df = (master_df[master_df["group"].isin(groups_to_plot) &
                              ~master_df["Mouse"].isin(mice_to_exclude)] 
                    .copy())
    processed_df["run"] = processed_df["run"].astype(str)

    # Loop through these to create graphs
    metrics_to_plot = ["void", "leak", "Avg Void Vol (ul)"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(9,6))

        # Create a map for x-axis jitter for scatter plot and x position from runs_to_include
        unique_mice = processed_df["Mouse"].unique()
        jitter_map = {mouse: (i - len(unique_mice) / 2) * 0.05 for i, mouse in enumerate(unique_mice)}
        run_map = {val: i for i, val in enumerate(runs_to_include)}

        processed_df["x_numeric"] = processed_df["run"].map(run_map) + processed_df["Mouse"].map(jitter_map)
        processed_df["Legend Label"] = processed_df.apply(
            lambda row: f"{row["Mouse"]} ({row["cohort"]})",
            axis=1
        )

        # Plot
        sns.barplot(
            data=processed_df,
            x="run",
            y=metric,
            order=runs_to_include,
            alpha=0.3, 
            width=0.6,
            gap=0,
            color="gray",
            capsize=0.05
        )
        sns.lineplot(
            data=processed_df,
            x="x_numeric",
            y=metric,
            units="Mouse",
            estimator=None,
            color="gray",
            alpha=0.4,
            linewidth=1,
        )   
        sns.scatterplot(
            data=processed_df,
            x="x_numeric",
            y=metric,
            hue="Legend Label",
            palette="tab10",
            s=40,
            edgecolor="gray",
            linewidth=1
        )
        
        # Plot settings
        plt.title(f"{plot_title}: {metric.upper()}")
        plt.legend(title="Mice", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()

        # Save plot
        plot_file_name = save_path / f"{metric.upper()}.png"
        plt.savefig(plot_file_name, dpi=300, bbox_inches="tight")
        print(f"Saved {plot_file_name}")
        # plt.show()

    # Save processed_df 
    df_save_path = save_path / f"{plot_title}_plot_data.csv"
    processed_df.to_csv(df_save_path, index=True)
    print(f"Saved {df_save_path}")