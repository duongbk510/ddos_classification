import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle
# from pandas_ml import ConfusionMatrix

# read data for training
df = pd.read_csv('data_train.csv')

# read data for testing
df_test = pd.read_csv('data_test.csv')

# encoding
le = LabelEncoder()
df['PKT_TYPE'] = le.fit_transform(df['PKT_TYPE'])
# ,'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO', 'PKT_CLASS'])
df['FLAGS'] = le.fit_transform(df['FLAGS'])
df['NODE_NAME_FROM'] = le.fit_transform(df['NODE_NAME_FROM'])
df['NODE_NAME_TO'] = le.fit_transform(df['NODE_NAME_TO'])
# df['PKT_CLASS'] = le.fit_transform(df['PKT_CLASS'])
df_test['PKT_TYPE'] = le.fit_transform(df_test['PKT_TYPE'])
# ,'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO', 'PKT_CLASS'])
df_test['FLAGS'] = le.fit_transform(df_test['FLAGS'])
df_test['NODE_NAME_FROM'] = le.fit_transform(df_test['NODE_NAME_FROM'])
df_test['NODE_NAME_TO'] = le.fit_transform(df_test['NODE_NAME_TO'])
# config param for model
model_name = 'rf_clf_2.sav'
n_trees = 50
labels = ['Normal', 'UDP-Flood', 'HTTP-FLOOD', 'SIDDOS', 'Smurf' ]

# select feature and label
X_train = df.iloc[:,:27]
y_train = df.iloc[:,27]

X_test = df_test.iloc[:,:27]
y_test = df_test.iloc[:,27]

rf  = RandomForestClassifier(n_estimators = n_trees)
rf.fit(X_train, y_train)

# save model
pickle.dump(rf, open(model_name, 'wb'))

# load model
model_clf = pickle.load(open(model_name, 'rb'))

# show result
predict_train = model_clf.predict(X_train)
predict_test = model_clf.predict(X_test)
cmtx = pd.DataFrame(confusion_matrix(y_test,predict_test, labels=labels), index=labels, columns=labels)

print('Accuracy of RF classifier on training set: {:.4f}'.format(model_clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.4f}'.format(model_clf.score(X_test, y_test)))
print('confusion matrix:')
print(cmtx)