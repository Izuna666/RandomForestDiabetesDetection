import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------getting the data ready----------------------------------------------
Data = pd.read_csv("Ready_Iraq_dataset.csv")
Data0 = Data[Data['CLASS'] == 0].sample(n=53,random_state=42, ignore_index=False)
Data1 = Data[Data['CLASS'] == 1].sample(n=53,random_state=42, ignore_index=False)
Data2 = Data[Data['CLASS'] == 2].sample(n=53,random_state=42, ignore_index=False)

Data53Samples = pd.concat([Data0, Data1, Data2], axis=0)
#ready data with 53 samples
Data53Samples = Data53Samples.reset_index(drop=True)
NumericData = Data53Samples.drop(['CLASS'], axis = 1, inplace = True)   #dropping class
# making sure we use only numeric values even when our dataset is good
NumericData = Data53Samples.select_dtypes(include=['number'])
# if anything got dropped we only use relevant data and names
columns = NumericData.columns

# Creating empty DataFrames for correlation coefficients and p-values
SpearmanCorr = pd.DataFrame(index=columns, columns=columns)
p_values = pd.DataFrame(index=columns, columns=columns)

#----------------------------------Computing Spearman correlation and p-values-----------------------------------
for col1 in columns:
    for col2 in columns:
        corr, p = spearmanr(NumericData[col1], NumericData[col2])
        SpearmanCorr.loc[col1, col2] = corr
        p_values.loc[col1, col2] = p

# Converting to float since dataframes have NaN values
SpearmanCorr = SpearmanCorr.astype(float)
p_values = p_values.astype(float)

# --------------------------------------------Displaying the data-----------------------------------------------
#print("Spearman Correlation Matrix:")
#print(SpearmanCorr.round(2))

#significant_mask = p_values < 0.05
#print("\n Statistically Significant Correlations (p < 0.05):")
#print(SpearmanCorr.where(significant_mask).round(2))

#---------------------------------------------ploting heatmap---------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(SpearmanCorr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Spearman Correlation Heatmap")
plt.show()

