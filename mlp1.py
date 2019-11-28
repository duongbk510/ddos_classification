import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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
# svm.fit(X_train, y_train)
mlp.fit(X_train, y_train)
# print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
print('Accuracy of MLP classifier on training set: {:.2f}'.format(mlp.score(X_train, y_train)))
print('Accuracy of MLP classifier on test set: {:.2f}'.format(mlp.score(X_test, y_test)))

# import pandas as pd
# import numpy as np 
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn.neural_network import MLPClassifier
# from sklearn.neural_network import MLPRegressor

# # Import necessary modules
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# from sklearn.metrics import r2_score

# df = pd.read_csv('diabetes.csv') 
# print(df.shape)
# df.describe().transpose()

# target_column = ['diabetes'] 
# predictors = list(set(list(df.columns))-set(target_column))
# df[predictors] = df[predictors]/df[predictors].max()
# df.describe().transpose()

# X = df[predictors].values
# y = df[target_column].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
# print(X_train.shape); print(X_test.shape)

# from sklearn.neural_network import MLPClassifier

# mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
# mlp.fit(X_train,y_train)

# predict_train = mlp.predict(X_train)
# predict_test = mlp.predict(X_test)

# from sklearn.metrics import classification_report,confusion_matrix
# print(confusion_matrix(y_train,predict_train))
# print(classification_report(y_train,predict_train))

# print(confusion_matrix(y_test,predict_test))
# print(classification_report(y_test,predict_test))