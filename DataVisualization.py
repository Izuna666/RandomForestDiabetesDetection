import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#using our prepared data
Data = pd.read_csv("Ready_Iraq_dataset.csv")

Data_0 = Data[Data['CLASS'] == 0].sample(n=53, ignore_index=False)
Data_1 = Data[Data['CLASS'] == 1].sample(n=53, ignore_index=False)
Data_2 = Data[Data['CLASS'] == 2].sample(n=53, ignore_index=False)

#function for easy display

# def  PlotHistogram(x,y, DataName,column, giveLabel, giveBins, giveAlpha):
#     for x in DataName:
#         plt.hist(DataName[column], label = giveLabel, bins = giveBins, alpha = giveAlpha)
#         plt.show()

# PlotHistogram(0,0,Data_0,"AGE",'Healthy',12,0.7)
