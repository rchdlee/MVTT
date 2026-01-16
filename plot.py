import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

# Read master csv file
root_dir = Path(r"")

master_csv_path = list(root_dir.glob("MASTERDATA*"))[0]

master_df = pd.read_csv(master_csv_path)

# Filter master by cohort(s) to be analyzed
groups_to_plot = ["AdultF"]
x_axis = "run" # run -> single cohort; cohort -> multiple cohorts
y_axis = "void"
# y_axis = "leak"

plot_df = master_df[master_df["group"].isin(groups_to_plot)].copy()
plot_df[x_axis] = plot_df[x_axis].astype(str)
plot_df["Legend Label"] = plot_df.apply(
    lambda row: f"{row["Mouse"]} ({row["group"]})",
    axis=1
)

unique_mice = plot_df["Mouse"].unique()
jitter_map = {mouse: (i - len(unique_mice) / 2) * 0.05 for i, mouse in enumerate(unique_mice)}

run_map = {"1": 0, "2": 1}
plot_df["x_numeric"] = plot_df[x_axis].map(run_map) + plot_df["Mouse"].map(jitter_map)

# run
run_order = sorted(plot_df[x_axis].unique())

print(plot_df, run_order)
plt.figure(figsize=(7,6))

sns.barplot(
    data=plot_df, 
    x=x_axis, 
    y=y_axis,
    order=run_order,
    alpha=0.3,
    width=0.5,
    gap=0,
    color="gray",
    capsize=0.05
    )
sns.lineplot(
    data=plot_df,
    # x=x_axis,
    x="x_numeric", 
    y=y_axis,
    units="Mouse",
    estimator=None,
    color="gray",
    alpha=0.4,
    linewidth=1,
)
sns.scatterplot(
    data=plot_df, 
    # x=x_axis, 
    x="x_numeric", 
    y=y_axis,
    # hue="Mouse",
    hue="Legend Label",
    palette="tab10",
    s=50,
    edgecolor="gray",
    linewidth=1,
    )


plt.legend(title="Mice", bbox_to_anchor=(1.05, 1), loc="upper left",)
# plt.xticks(ticks=[0,1], labels=run_order)
plt.title(f"{", ".join(groups_to_plot)}")
plt.xlim(-0.7, 1.7)
plt.tight_layout()
plt.show()
