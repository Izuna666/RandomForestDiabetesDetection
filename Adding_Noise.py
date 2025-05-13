import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------using our prepared data-------------------------------------------------
data = pd.read_csv("Ready_Iraq_dataset.csv")
#spliting our data to three categories: healthy=0, maybe=1, diabetes=2
Data_0 = data[data['CLASS'] == 0].sample(n=53,random_state=42, ignore_index=False)
Data_1 = data[data['CLASS'] == 1].sample(n=53,random_state=42, ignore_index=False)
Data_2 = data[data['CLASS'] == 2].sample(n=53,random_state=42, ignore_index=False)

used_columns = data.columns.drop(["Gender","CLASS"])                                            #we take all columns except gender and class because we will not calculate the deviation from it

print(data.head())
#-------------------calculatin mean and standard deviation for noise simulation-------------------------------
sigma = data[used_columns].std()                                                                #std returns sample standard deviation over requested axis.
noise = np.random.normal(loc=0.0,scale=sigma.values,size=(data.shape[0],len(used_columns)))     #Draw random samples from a normal (Gaussian) distribution.

#print(noise)
#--------------------applying the noise and making a new noisy dataset-----------------------------------------
noisy_data = data.copy()
noisy_data[used_columns] +=(noise*0.2)                                                          #making noise smaller

print(noisy_data.head())
noisy_data.to_csv("Noisy_Iraq_dataset.csv", index = False)                                     #making new dataset

# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Subplot 1: Bar chart of sigmas per feature
ax1.bar(used_columns, sigma)
ax1.set_xlabel('Feature')
ax1.set_ylabel('Standard Deviation')
ax1.set_title('Feature Standard Deviations')
ax1.tick_params(axis='x', rotation=45)

# --- Subplot 2: Histogram of sigma values
ax2.hist(sigma, bins='auto')
ax2.set_xlabel('Standard Deviation')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Feature Standard Deviations')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()