import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from pathlib import Path


def plot_averaged_data(root_dir: Path, groups_to_plot: list, comparison_name: str, output_dir="plots"): # would be nice to add a variable compare_within_cohort = False 
    # Setup directories
    today = date.today().strftime("%Y-%m-%d")
    save_path = Path(root_dir / "PLOTS" / f"{today}_{comparison_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load MASTERDATA csv as master_df
    master_csv_path = list(root_dir.glob("MASTERDATA*"))[0]
    master_df = pd.read_csv(master_csv_path)

    # Filter master_df by groups_to_plot
    plot_df = master_df[master_df["group"].isin(groups_to_plot)].copy()
    
    # Bool to deterimine whether to plot Run1 vs Run2 or cohorts
    is_single_cohort = len(groups_to_plot) == 1

    # Process Data
    if is_single_cohort:
        processed_df = plot_df
        processed_df["run"] = processed_df["run"].astype(str)

        x_axis = "run"
        metrics_to_plot = ["void", "leak", "Avg Void Vol (ul)"]
    else:
        processed_df = plot_df.groupby(["Mouse", "group"]).apply(
            lambda x: pd.Series({
                "void": x["void"].mean(),
                "leak": x["leak"].mean(),
                "wavv": (x["Avg Void Vol (ul)"] * x["void"]).sum() / x["void"].sum() if x["void"].sum() > 0 else np.nan
            }),
            include_groups=False
        ).reset_index()

        x_axis = "group"
        metrics_to_plot = ["void", "leak", "wavv"]

    # Save processed data as csv
    csv_file_name = save_path / f"{comparison_name}_source_data.csv"
    processed_df.to_csv(csv_file_name, index=False)
    print(f"Data saved to: {csv_file_name}")

    # Generate and save plots
    print(f"metrics to plot: {metrics_to_plot}")
    for metric in metrics_to_plot:
        plt.figure(figsize=(7,6))

        if is_single_cohort:
            # Setup varaibles
            run_order = sorted(processed_df[x_axis].unique())
            processed_df[x_axis] = processed_df[x_axis].astype(str)
            # Scatter point jitter logic
            unique_mice = processed_df["Mouse"].unique()
            jitter_map = {mouse: (i - len(unique_mice) / 2) * 0.025 for i, mouse in enumerate(unique_mice)}
            run_map = {val: i for i, val in enumerate(run_order)}
            processed_df["x_numeric"] = processed_df[x_axis].map(run_map) + processed_df["Mouse"].map(jitter_map)
            # Legend Label
            processed_df["Legend Label"] = processed_df.apply(
                lambda row: f"{row["Mouse"]} ({row["cohort"]})",
                axis=1
                )
            print("starting to plot\n", processed_df)
            # Plot
            sns.barplot(
                data=processed_df, 
                x=x_axis, 
                y=metric,
                order=run_order,
                alpha=0.3,
                width=0.6,
                gap=0,
                color="gray",
                capsize=0.05
                )
            sns.lineplot(
                data=processed_df,
                # x=x_axis,
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
                # x=x_axis, 
                x="x_numeric", 
                y=metric,
                # hue="Mouse",
                hue="Legend Label",
                palette="tab10",
                s=50,
                edgecolor="gray",
                linewidth=1,
                )

        else: 
            sns.barplot(
            data=processed_df, 
            x=x_axis, 
            y=metric, 
            order=groups_to_plot,
            alpha=0.3, 
            palette='muted', 
            capsize=0.05)
            sns.stripplot(
                data=processed_df, 
                x=x_axis, 
                y=metric, 
                order=groups_to_plot, 
                # hue="group", 
                color="gray",
                size=6.5, 
                jitter=0.1, 
                edgecolor="gray", 
                linewidth=0.75,
                legend=False
                )

        # Formatting
        plt.title(f"{comparison_name.upper()}: {metric.upper()}")
        plt.ylim(bottom=0)


        if is_single_cohort:
            plt.legend(title="Mice", bbox_to_anchor=(1.05, 1), loc="upper left")
        if metric == "leak":
            current_max = processed_df["leak"].max()
            print(f"MAX IS {current_max}")
            if current_max < 10:
                plt.ylim(0, 10)
            else: 
                plt.ylim(0, current_max + 3)
        if metric == "wavv" or metric == "Avg Void Vol (ul)":
            plt.ylim(0, 750)  

        plt.tight_layout()

        # Save plot
        plot_file_name = save_path / f"{comparison_name}_{metric}.png"
        plt.savefig(plot_file_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_file_name}")

###


project_dir = Path(r"")
groups = ["TeenF", "AdultF", "MenoF", "Nulliparous", "Parous", "VCD", "TeenM", "AdultM", "OldMales"]
# groups = ["OldMales"]

def plot_every_individual_group():
    for group in groups:
        plot_averaged_data(project_dir, [group], group.capitalize())

plot_every_individual_group()

# plot_averaged_data(project_dir, ["Nulliparous", "Parous", "VCD", "MenoF", "AdultF", "TeenF", ], "Comparing Female Cohorts")
# plot_averaged_data(project_dir, ["TeenM", "AdultM", "OldMales"], "Comparing Male Cohorts")
# plot_averaged_data(project_dir, ["TeenF", "AdultF", "MenoF", "Nulliparous", "Parous", "VCD", "TeenM", "AdultM", "OldMales"], "Comparing All Cohorts")
