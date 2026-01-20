import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from datetime import date
from pathlib import Path


def plot_averaged_data(root_dir: Path, groups_to_plot: list, comparison_name: str): # would be nice to add a variable compare_within_cohort = False 


    # Setup directories
    is_single_group = len(groups_to_plot) == 1
    if is_single_group:
        prefix = "SINGLE"
    if not is_single_group:
        prefix = "MULTI"
    today = date.today().strftime("%Y-%m-%d")
    save_path = Path(root_dir / "__PLOTS__" / f"{prefix}_{today}_{comparison_name.upper()}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load MASTERDATA csv as master_df
    master_csv_path = list(root_dir.glob("MASTERDATA*"))[0]
    master_df = pd.read_csv(master_csv_path)
    print("MASTERDF:",master_df)

    # 1. Check the data types
    print(f"Column Type: {master_df['group'].dtype}")
    print(f"List Item Type: {type(groups_to_plot[0])}")

    # 2. Look for hidden spaces (using repr() to show quotes and spaces)
    print("Actual values in Column:", master_df["group"].unique()[:5])
    print("Requested values in List:", groups_to_plot)

    # Filter master_df by groups_to_plot
    plot_df = master_df[master_df["group"].isin(groups_to_plot)].copy()
    print("\n\nPLOTDF", plot_df)
    
    # Bool to deterimine whether to plot Run1 vs Run2 or groups
    is_single_group = len(groups_to_plot) == 1

    # Process Data
    if is_single_group:
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
    for metric in metrics_to_plot:
        plt.figure(figsize=(7,6))

        # Plot Run 1 vs Run 2 for a single group
        if is_single_group:
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

        # Plot groups in groups_to_plot and run stats
        else: 
            # Prepare data
            groups_data = [processed_df[processed_df["group"] == g][metric].dropna() for g in groups_to_plot]
            
            #===================
            # Non-Parametric 
            #-------------------
            # Kruskal-Wallis H-test
            h_stat, p_val = stats.kruskal(*groups_data)
            sig_label = f"Kruskal-Wallis p={p_val:.4f}"

            significant_pairs = []

            if p_val < 0.05:
                print(f"Significant Differnces found in {metric} (p={p_val:.4f})")

                # Dunn's post-hoc test
                p_matrix = sp.posthoc_dunn(processed_df, val_col=metric, group_col="group", p_adjust="holm")
                print(f"P-MATRIX FOR {metric.upper()}:\n{p_matrix}")
                p_matrix_save_path = save_path / f"{comparison_name}_{metric.upper()}_dunn_matrix.csv"
                p_matrix.to_csv(p_matrix_save_path)
                for i, g1 in enumerate(groups_to_plot):
                    for g2 in groups_to_plot[i+1:]:
                        p_adj = p_matrix.loc[g1, g2]
                        if p_adj < 0.05:
                            significant_pairs.append({"group1": g1, "group2": g2, "p-adj": p_adj})

            if significant_pairs:
                sig_df = pd.DataFrame(significant_pairs)
                sig_save_path = save_path / f"{comparison_name}_{metric.upper()}_significant_pairs.csv"
                sig_df.to_csv(sig_save_path, index=False)

            #-Brackets
            # Increase fig height by 0.5" / bracket
            num_sig = len(significant_pairs)
            if num_sig > 2:
                plt.gcf().set_size_inches(7, 6 + (num_sig * 0.5))

            # Draw brackets
            y_max = processed_df[metric].max()
            line_offset = y_max * 0.08
            line_height = y_max * 0.03

            print(significant_pairs, y_max, processed_df)
            stack_count = 0

            for pair in significant_pairs:
                x1 = groups_to_plot.index(pair["group1"])
                x2 = groups_to_plot.index(pair["group2"])

                stack_count += 1
                level = y_max + (stack_count + 1) * line_offset

                plt.plot([x1, x1, x2, x2], [level-line_height, level, level, level-line_height], color="black", linewidth=1.2)

                label = "*" if pair["p-adj"] < 0.05 else ""
                if pair["p-adj"] < 0.01: label = "**"
                if pair["p-adj"] < 0.001: label = "***"

                plt.text((x1 + x2) / 2, level, label, ha="center", va="bottom", color="black")

            # #===================
            # # Parametric 
            # #-------------------
            # # ANOVA
            # f_stat, p_val = stats.f_oneway(*groups_data)
            # y_max = processed_df[metric].max()

            # sig_label = f"ANOVA p={p_val:.4f}"

            # if p_val < 0.05:
            #     tukey = pairwise_tukeyhsd(endog=processed_df[metric].dropna(), groups=processed_df["group"].dropna(), alpha=0.05)
            #     print(tukey)
            #     tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            #     significant_pairs = tukey_df[tukey_df["reject"] == True]

            #     # Save tukey df
            #     tukey_df_save_path = save_path / f"{metric.upper()}_TukeyHSD.csv"
            #     tukey_df.to_csv(tukey_df_save_path, index=False)

            #     # Increase fig height by 0.5" / bracket
            #     num_sig = len(significant_pairs)
            #     if num_sig > 2:
            #         plt.gcf().set_size_inches(7, 6 + (num_sig * 0.5))

            #     # Draw brackets
            #     line_offset = y_max * 0.08
            #     line_height = y_max * 0.03

            #     print(significant_pairs, y_max)
            #     stack_count = 0

            #     for _, row in significant_pairs.iterrows():
            #         group1, group2 = row['group1'], row['group2']
            #         p_adj = row["p-adj"]

            #         x1 = groups_to_plot.index(group1)
            #         x2 = groups_to_plot.index(group2)

            #         stack_count += 1
            #         level = y_max + (stack_count + 1) * line_offset

            #         plt.plot([x1, x1, x2, x2], [level-line_height, level, level, level-line_height], color="black", linewidth=1.2)

            #         label = "*" if p_adj < 0.05 else ""
            #         if p_adj < 0.01: label = "**"
            #         if p_adj < 0.001: label = "***"

            #         plt.text((x1 + x2) / 2, level, label, ha="center", va="bottom", color="black")

            sns.barplot(
                data=processed_df, 
                x=x_axis, 
                y=metric, 
                order=groups_to_plot,
                alpha=0.3, 
                hue=x_axis,
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

        if is_single_group:
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

            # if not is_single_group:
            #     plt.ylim(0, y_max * 1.05 + (len(significant_pairs) + 2) * line_offset)

        if not is_single_group:
            plt.title(f"{comparison_name.upper()}: {metric.upper()}\n{sig_label}")
            plt.ylim(0, y_max * 1.05 + (len(significant_pairs) + 2) * line_offset)



        plt.tight_layout()



        # Save plot
        plot_file_name = save_path / f"{comparison_name}_{metric}.png"
        plt.savefig(plot_file_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {plot_file_name}")

        # DISTRIBUTION PLOT
        if is_single_group:
            plt.figure(figsize=(7,6))

            dist_summary = processed_df.groupby(["Mouse", "cohort"])[metric].mean().reset_index()

            sns.kdeplot(data=dist_summary, x=metric, fill=True, alpha=0.2, bw_adjust=0.7)
            sns.rugplot(data=dist_summary, x=metric, hue="cohort", height=0.1, linewidth=2)
            
            plt.title(f"Check for Normality (N={len(dist_summary)} mice)")\

            # plt.title(f"DISTRIBUTION: {comparison_name.upper()} - {metric.upper()}")
            plt.xlabel(f"{metric}")
            plt.ylabel(f"Frequency")

            dist_file_name = save_path / f"{comparison_name}_{metric}_distribution.png"
            plt.savefig(dist_file_name, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Distribution plot saved to: {dist_file_name}")



def plot_sequential_data(root_dir: Path, plot_title:str, groups_to_plot: list, mice_to_exclude: list, runs_to_exclude: list):
    # Setup directories
    today = date.today().strftime("%Y-%m-%d")
    save_path = Path(root_dir / "__PLOTS__" / f"{today}_{groups_to_plot[0]}")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load MASTERDATA csv as master_df
    master_csv_path = list(root_dir.glob("MASTERDATA*"))[0]
    master_df = pd.read_csv(master_csv_path)

    # Filter master_df by groups_to_plot
    plot_df = master_df[master_df["group"].isin(groups_to_plot)].copy()
    print(plot_df)
    
    # Bool to deterimine whether to plot Run1 vs Run2 or groups
    is_single_group = len(groups_to_plot) == 1


    mouse_filtered_df = plot_df[~plot_df["Mouse"].isin(mice_to_exclude)]
    mouse_run_filtered_df = mouse_filtered_df[~plot_df["run"].isin(runs_to_exclude)]
    processed_df = mouse_run_filtered_df.copy()
    processed_df["run"] = processed_df["run"].astype(str)

    print(processed_df)
    x_axis = "run"
    metrics_to_plot = ["void", "leak", "Avg Void Vol (ul)"]


    for metric in metrics_to_plot:
        plt.figure(figsize=(9,6))

        # Plot Run 1 vs Run 2 for a single group
        if is_single_group:
            # Setup varaibles
            run_order = sorted(processed_df[x_axis].unique())
            processed_df[x_axis] = processed_df[x_axis].astype(str)
            # Scatter point jitter logic
            unique_mice = processed_df["Mouse"].unique()
            jitter_map = {mouse: (i - len(unique_mice) / 2) * 0.05 for i, mouse in enumerate(unique_mice)}
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
                s=40,
                edgecolor="gray",
                linewidth=1,
                )

            plt.title(f"{plot_title}:\n {metric.upper()}")
            plt.legend(title="Mice", bbox_to_anchor=(1.05, 1), loc="upper left")

            if metric == "void":
                plt.ylim([0,10])
                plt.ylabel(f"{metric.capitalize()} count")
            if metric == "leak":
                plt.ylim([0,20])
            if metric == "Avg Void Vol (ul)":
                plt.ylim([0,1200])
                # plt.ylim([0,750])
                plt.ylabel(f"{metric.capitalize()}")


            plt.tight_layout()
            plot_file_name = save_path / f"SEQUENTIAL_{plot_title}_{metric}.png"
            plt.savefig(plot_file_name, dpi=300, bbox_inches="tight")
            # plt.show()

#======================================================================================================================
#======================================================================================================================
#======================================================================================================================

#########################################################################
project_dir = Path(r"C:\Users\Richard\_Vork\MVT\data\averaged_runs_new")
#########################################################################

groups = ["TeenF", "AdultF", "MenoF", "Nulliparous", "Parous", "VCD", "TeenM", "AdultM", "OldMales"]
# groups = ["OldMales"]

def plot_every_individual_group():
    for group in groups:
        plot_averaged_data(project_dir, [group], group.capitalize())

plot_every_individual_group()
plot_averaged_data(project_dir, ["TeenF", "AdultF", "MenoF", "Nulliparous", "Parous", "VCD", "TeenM", "AdultM", "OldMales"], "Comparing All Cohorts")

plot_averaged_data(project_dir, ["Nulliparous", "Parous", "VCD", "MenoF", "AdultF", "TeenF", ], "Comparing Female Cohorts")
plot_averaged_data(project_dir, ["TeenM", "AdultM", "OldMales"], "Comparing Male Cohorts")
plot_averaged_data(project_dir, ["Nulliparous", "Parous", "MenoF"], "Parity")
plot_averaged_data(project_dir, ["Nulliparous", "Parous", "MenoF", "VCD"], "Old Female Mice")
plot_averaged_data(project_dir, ["Nulliparous", "Parous", "MenoF"], "Parity")
plot_averaged_data(project_dir, ["TeenF", "TeenM"], "TeenF vs. TeenM")
plot_averaged_data(project_dir, ["AdultF", "AdultM"], "AdultF vs. AdultM")
plot_averaged_data(project_dir, ["AdultF", "MenoF", "Nulliparous", "Parous", "VCD"], "Females without TeenF")

# #########################################################################
# project_dir = Path(r"C:\Users\Richard\_Vork\MVT\ERa_KO")
# #########################################################################

# ##plotsequentialdata: project dir,
# # plot_sequential_data(project_dir, "ERa-KO Females - ALL mice including miss", ["ELF"], [], [])
# # plot_sequential_data(project_dir, "ERa-KO Females - only bilateral hits", ["ELF"], [4203, 4260, 4990, 4991, 4988], [])
# # plot_sequential_data(project_dir, "ELF1 Hits", ["ELF"], [])

# # ELM1 || ELM2
# # 3658 || 4495
# # 3659 || 4496
# # 3524 || 4497
# # 3525 || 4470

# plot_sequential_data(project_dir, "ELM1", ["ELM"], [4495, 4496, 4497, 4470], [])
# plot_sequential_data(project_dir, "ELM2", ["ELM"], [3658, 3659, 3524, 3525, 4496, 4470], [])
# plot_sequential_data(project_dir, "ERa-KO Males - only bilateral hits", ["ELM"], [4496, 4470], [])
# plot_sequential_data(project_dir, "ERa-KO Males - Control", ["ELM"], [3658, 3659, 3524, 3525, 4495, 4496, 4497], [])