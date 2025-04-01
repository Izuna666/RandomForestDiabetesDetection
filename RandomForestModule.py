import pandas as pd
import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import matplotlib.pyplot as plt

Data = pd.read_csv("Ready_Iraq_dataset.csv")
#Data["CLASS"] = Data.target

X = Data.iloc[:,:-1].values
Y = Data.iloc[:, -1].values

#print(Y)
#print(X)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=1)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

conf_matrix = sklearn.metrics.confusion_matrix(Y_test, Y_pred)
display_matrix = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = conf_matrix)
display_matrix.plot()
plt.show()