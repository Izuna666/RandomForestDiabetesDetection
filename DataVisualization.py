import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------using our prepared data-------------------------------------------------
Data = pd.read_csv("Ready_Iraq_dataset.csv")
#spliting our data to three categories: healthy=0, maybe=1, diabetes=2
Data_0 = Data[Data['CLASS'] == 0].sample(n=53,random_state=42, ignore_index=False)
Data_1 = Data[Data['CLASS'] == 1].sample(n=53,random_state=42, ignore_index=False)
Data_2 = Data[Data['CLASS'] == 2].sample(n=53,random_state=42, ignore_index=False)

#----------------------------------all our funtions--------------------------------------------------------------------
# Reusable function for plotting histogram for a given column
def PlotHistogram(ax, col, datasets, labels, colors, bins=12, alpha=0.7, xlabel=None, ylabel=None, title=None):

    for data, label, color in zip(datasets, labels, colors):
        ax.hist(data[col], label=label, bins=bins,color=color, alpha=alpha)
    ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
# Reusable function for plotting histogram for a given column with set bias and max min on x so the dataset is form the same series
def PlotHistogramBias(ax, col, datasets, labels, colors, bins=12, alpha=0.7, xlabel=None, ylabel=None, title=None):

    # Combine data from all datasets for the specific column
    all_values = np.concatenate([data[col].values for data in datasets])
    
    # Calculate the overall minimum and maximum of the combined data
    x_min = all_values.min()
    x_max = all_values.max()
    
    # Create bins using the same range for all datasets
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    
    # Plot each dataset's histogram using the same bins
    for data, label, color in zip(datasets, labels, colors):
        ax.hist(data[col], label=label, bins=bin_edges, color=color, alpha=alpha)
    
    # Set consistent x-axis limits for comparison
    ax.set_xlim(x_min, x_max)
    
    ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
#adding weights to our data so we have percentage of it
def PlotHistogramPercent(ax, col, datasets, labels, colors, bins=12, alpha=0.7, xlabel=None, ylabel=None, title=None):

    # Combine data across all datasets to get uniform bin edges and x-limits
    all_values = np.concatenate([data[col].values for data in datasets])
    x_min = all_values.min()
    x_max = all_values.max()
    bin_edges = np.linspace(x_min, x_max, bins + 1)
    
    # Plot each dataset's histogram with normalized weights (percentage)
    for data, label, color in zip(datasets, labels, colors):
        # Calculate weights so that sum(weights) equals 100 for each dataset.
        values = data[col].values
        weights = np.ones_like(values) * (100.0 / len(values))
        ax.hist(values, label=label, bins=bin_edges, color=color, alpha=alpha, weights=weights)
    
    # Set the x-axis limits and apply other formatting.
    ax.set_xlim(x_min, x_max)
    ax.legend()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        # Change label to indicate percentages.
        ax.set_ylabel(ylabel if ylabel else "Percentage (%)")
    if title:
        ax.set_title(title)
#funtion for removing blank subplots
def RemovePlot(axes,columns):
    # If there are extra subplot spaces, remove them.
    if len(axes) > len(columns):
        for ax in axes[len(columns):]:
            ax.remove()

#----------------------all our lists with difrent data for our funtions to go through------------------------------------
columns = ['Gender', 'AGE', 'Urea', 'Cr', 'HbA1c', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']
datasets = [Data_0, Data_1, Data_2]
labels = ['Healthy', 'Maybe', 'Diabetes']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

#--------------------------------displaying our histograms with our funtions---------------------------------------------
fig1, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()

for ax, col in zip(axes, columns):
    PlotHistogram(ax, col, datasets, labels,colors, bins=12, alpha=0.7, xlabel=col, ylabel="Frequency", title=f"{col} Histogram")

RemovePlot(axes=axes,columns=columns)
plt.tight_layout()
plt.show()
plt.close(fig1)

#--------------------------------plot with uniform x---------------------------------------------------------------------
fig2, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()
# Loop over the columns and plot histograms for each using our function.
for ax, col in zip(axes, columns):
    PlotHistogramBias(ax, col, datasets, labels, colors, bins=12, alpha=0.6, xlabel=col, ylabel="Frequency", title=f"{col} Histogram")

RemovePlot(axes=axes,columns=columns)
plt.tight_layout()
plt.show()
plt.close(fig2)

#------------------------plot with percantage-------------------------------------------------------------------------------------
fig3, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
axes = axes.flatten()

for ax, col in zip(axes, columns):
    PlotHistogramPercent(ax, col, datasets, labels, colors, bins=12, alpha=0.7, xlabel=col, ylabel="Percentage (%)", title=f"{col} Histogram")

RemovePlot(axes=axes,columns=columns)
plt.tight_layout()
plt.show()
plt.close(fig3)