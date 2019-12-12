import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report,confusion_matrix

# read data
# data = arff.loadarff('final dataset.arff')
# df = pd.DataFrame(data[0])
df_test = pd.read_csv('data_test.csv')

# feature encoding
le = LabelEncoder()
df_test['PKT_TYPE'] = le.fit_transform(df_test['PKT_TYPE'])
df_test['FLAGS'] = le.fit_transform(df_test['FLAGS'])
df_test['NODE_NAME_FROM'] = le.fit_transform(df_test['NODE_NAME_FROM'])
df_test['NODE_NAME_TO'] = le.fit_transform(df_test['NODE_NAME_TO'])
# df['PKT_CLASS'] = le.fit_transform(df['PKT_CLASS'])

# config param and model
labels = ['Normal', 'UDP-Flood', 'HTTP-FLOOD', 'SIDDOS', 'Smurf' ]
# list_model_name = ['rf_clf_1.sav', 'mlp_clf_1.sav', 'svm_clf_1.sav']
model_name = 'rf_clf_2.sav'
# split dataset
X = df_test.iloc[:,:27]
y = df_test.iloc[:,27]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)


# load model
model_clf = pickle.load(open(model_name, 'rb'))

# show results
# predict_train = model_clf.predict(X_train)
predict_test = model_clf.predict(X)
cmtx = pd.DataFrame(confusion_matrix(y,predict_test, labels=labels), index=labels, columns=labels)

# print('Accuracy of SVM classifier on training set: {:.2f}'.format(model_clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.4f}'.format(model_clf.score(X, y)))

# print(confusion_matrix(y_test,predict_test))
# print(classification_report(y_test,predict_test))
print('confusion matrix:')
print(cmtx)
