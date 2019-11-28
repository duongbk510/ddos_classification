import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

data = arff.loadarff('final dataset.arff')
df = pd.DataFrame(data[0])

le = LabelEncoder()
df['PKT_TYPE'] = le.fit_transform(df['PKT_TYPE'])
# ,'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO', 'PKT_CLASS'])
df['FLAGS'] = le.fit_transform(df['FLAGS'])
df['NODE_NAME_FROM'] = le.fit_transform(df['NODE_NAME_FROM'])
df['NODE_NAME_TO'] = le.fit_transform(df['NODE_NAME_TO'])
df['PKT_CLASS'] = le.fit_transform(df['PKT_CLASS'])

X = df.iloc[:,:27]
y = df.iloc[:,27]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

# svm = SVC()
mlp = MLPClassifier(hidden_layer_sizes=(16), learning_rate=0.3, momentum=0.2, max_iter=500)
rf  = RandomForestClassifier(n_estimators = 50)
# svm.fit(X_train, y_train)
# mlp.fit(X_train, y_train)
rf.fit(X_train, y_train)
# print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
# print('Accuracy of MLP classifier on training set: {:.2f}'.format(mlp.score(X_train, y_train)))
# print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test, y_test)))
print('Accuracy of RF classifier on training set: {:.2f}'.format(rf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))