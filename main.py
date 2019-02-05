import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

csv = pd.read_csv('creditcard.csv')
#Print first five values in the dataset along with column ID
print(csv.head())

#To check if there are any NaN values in the dataset
#Checked individually from V1, V2, V3 to V28 and Amount with 0 NaN values
print("NaN values in the dataset:", sum(csv['Amount'].isna()))

print("Duplicate values in the dataset:", sum(csv.duplicated()))
csv.drop_duplicates(inplace=True)
print("Duplicate values removed")

#The panda datset is put into numpy array for computation in Sklearn
features = np.array(csv.iloc[:, 1:30])
labels = np.array(csv.iloc[:,30])

#Feature scaling using MinMaxScaler to ensure data lies in a certain range
print("Feature Scaling in progress")
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

#Feature selection using 24 principal components which give us the best value in classification metrics
print("Feature Selection in progress")
pca = PCA(n_components = 24)
features = pca.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print("Data split in training and testing dataset")

estimator = LogisticRegression(solver = 'sag', max_iter=17, tol=1, C=3)

print("Fitting data based on Logistic Regression")
estimator.fit(x_train, y_train)
print("Data Fitting complete")
y_pred = estimator.predict(x_test)

print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))