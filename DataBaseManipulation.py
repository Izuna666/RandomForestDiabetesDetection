import pandas as pd
import numpy as np

#This script was done once for a dataset to achive redable data:

Data = pd.read_csv("Iraq_dataset.csv")
Data['Gender'] = Data['Gender'].replace({'F':0, 'M':1})
Data['CLASS'] = Data['CLASS'].replace({'N':0, 'P':1, 'Y': 2})
Data.to_csv("Modified_Iraq_dataset.csv", index = False)

#now all datatypes are numerical so we can proceed:
#now we ge rid off unusefull columns:

Data = pd.read_csv("Modified_Iraq_dataset.csv")
Data.drop(['ID', 'No_Pation'], axis = 1, inplace = True)

Data.to_csv("Ready_Iraq_dataset.csv", index = False)
print(Data.dtypes)

#data is ready for use
