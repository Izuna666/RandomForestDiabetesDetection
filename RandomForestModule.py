'''
this script is OBSOLETE for newer versions of this project. Is only here for comparison and
archival purposes.
'''
import pandas as pd
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt
import numpy as np
import time

#Data = pd.read_csv("Ready_Iraq_dataset.csv")
Data = pd.read_csv("Noisy_Iraq_dataset.csv")
#-----------------------------extracting 53 samples---------------------------------
Data0 = Data[Data['CLASS'] == 0].sample(n=53,random_state=42, ignore_index=False)
Data1 = Data[Data['CLASS'] == 1].sample(n=53,random_state=42, ignore_index=False)
Data2 = Data[Data['CLASS'] == 2].sample(n=53,random_state=42, ignore_index=False)

Data53Samples = pd.concat([Data0, Data1, Data2], axis=0)
#ready data with 53 samples
Data53Samples = Data53Samples.reset_index(drop=True)

#------------------------making trainig data----------------------------------------
X1 = Data.iloc[:,:-1].values
Y1 = Data.iloc[:, -1].values

X2 = Data53Samples.iloc[:,:-1].values
Y2 = Data53Samples.iloc[:, -1].values

labels = [0,1,2]
#print(Y2)
#print(X2)
#--------------------building training models----------------------------------------
#added shuffle and stratify makes sure that our data is splited training/testing in the same ratio as ratio in class
X1_train, X1_test, Y1_train, Y1_test = sklearn.model_selection.train_test_split(X1, Y1, test_size=0.2, random_state=42, shuffle = True, stratify=Y1)
X2_train, X2_test, Y2_train, Y2_test = sklearn.model_selection.train_test_split(X2, Y2, test_size=0.2, random_state=42, shuffle = True, stratify=Y2)
#model_size = [10,20,40,60,80]           #liczba drzew
#model_size = [5,10,15,20,25]            #maxymalna glebokosc drzewa
model_size= [1,2,3,4,5]             #ilosc zmiennych branych pod uwage na najlepszy podzial
combined_all = []

for model in model_size:
    #timing our learning model
    start_timer = time.time()

    classifier1 = sklearn.ensemble.RandomForestClassifier(n_estimators=40, random_state=42, max_depth=15, max_features=model)
    classifier1.fit(X1_train, Y1_train)
    Y1_pred = classifier1.predict(X1_test)
    #//
    classifier2 = sklearn.ensemble.RandomForestClassifier(n_estimators=40, random_state=42, max_depth=15, max_features=model)
    classifier2.fit(X2_train, Y2_train)
    Y2_pred = classifier2.predict(X2_test)

    #------------------------testing our models------------------------------------------
    accuracy1 = sklearn.metrics.accuracy_score(Y1_test, Y1_pred)
    #print(f'Accuracy: {accuracy1 * 100:.2f}%')
    accuracy2 = sklearn.metrics.accuracy_score(Y2_test, Y2_pred)
    #print(f'Accuracy: {accuracy2 * 100:.2f}%')
    end_timer = time.time()
    training_time = end_timer - start_timer
    #//
    results1 = []
    for label in labels:
        precision1,recall1,f_score1,support1 = sklearn.metrics.precision_recall_fscore_support(Y1_test==label,Y1_pred==label)
        results1.append([label,recall1[0],recall1[1],precision1[1],f_score1[1],support1[1]])
    df1 = pd.DataFrame(results1,columns=["label","specifity","recall","precision","f_score","support"])
    print("test modelu zawierajacego " ,model, "drzew + ze wszystkimi danymi:\n",f'Accuracy: {accuracy1 * 100:.2f}%\n')
    results2 = []
    for label in labels:
        precision2,recall2,f_score2,support2 = sklearn.metrics.precision_recall_fscore_support(Y2_test==label,Y2_pred==label)
        results2.append([label,recall2[0],recall2[1],precision2[1],f_score2[1],support2[1]])
    df2 = pd.DataFrame(results2,columns=["label","specifity","recall","precision","f_score","support"])
    print("test modelu zawierajacego ",model, "drzew + z 53 samplami:\n",f'Accuracy: {accuracy2 * 100:.2f}%\n')
    print(f"Script took {training_time:.2f} seconds to run.")

    combined = pd.concat([df1, df2], ignore_index=True)
    combined_all.append(combined)

#//
final_df = pd.concat(combined_all, ignore_index=True)

#final_df.to_csv("merged_bejbi", index=False, float_format="%.3f")
#print(final_df)

#--------------------------------grafy na liczbe modeli--------------------------------------#
metrics = ["specifity", "recall", "precision", "f_score"]
models = ["All samples", "53 samples"]

# Add model column: Model1, Model2 alternating every 3 rows
final_df["model"] = [models[i // 3 % 2] for i in range(len(final_df))]

# Add trees column: repeat each tree value 6 times (3 for each model)
final_df["trees"] = [tree for tree in model_size for _ in range(6)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    for cls in labels:
        for model in models:
            subset = final_df[
                (final_df["label"] == cls) &
                (final_df["model"] == model)]
            ax.plot(subset["trees"], subset[metric], marker='o',
            label=f"{model} - Class {cls}",
            linewidth=2.5, alpha=0.8)
    
    ax.set_title(f"{metric.capitalize()}")
    ax.set_xlabel("ilosc cech")
    ax.set_ylabel("wartosc")
    ax.set_xticks(model_size)
    ax.grid(True, axis='x')
    ax.legend()

plt.suptitle("Dzialanie modeli w zaleznosci od ilosci cech", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#conf_matrix1 = sklearn.metrics.confusion_matrix(Y1_test, Y1_pred)
#conf_matrix2 = sklearn.metrics.confusion_matrix(Y2_test, Y2_pred)
##//
#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
#display_matrix1 = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix1)
#display_matrix2 = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix2)
#display_matrix1.plot(ax=axes[0])
#axes[0].set_title('Full Data classification')
#display_matrix2.plot(ax=axes[1])
#axes[1].set_title('53 Samples classification')
#plt.tight_layout()
#plt.show()

