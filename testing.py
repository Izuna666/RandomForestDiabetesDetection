#this is oly for testing and displaying our steps and milestones for a report
#it doesn't do anything new just for displaying things :)

import pandas as pd


Data = pd.read_csv("Iraq_dataset.csv")
data2 = pd.read_csv("Ready_Iraq_dataset.csv")


print("Baza danych przed modyfikacja: \n",Data.dtypes)
print("Baza danych po modyfikacji: \n", data2.dtypes)

#data is ready for use

