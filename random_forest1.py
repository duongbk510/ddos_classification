import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle

# read data
data = arff.loadarff('final dataset.arff')
df = pd.DataFrame(data[0])

# feature encoding
le = LabelEncoder()
df['PKT_TYPE'] = le.fit_transform(df['PKT_TYPE'])
# ,'FLAGS', 'NODE_NAME_FROM', 'NODE_NAME_TO', 'PKT_CLASS'])
df['FLAGS'] = le.fit_transform(df['FLAGS'])
df['NODE_NAME_FROM'] = le.fit_transform(df['NODE_NAME_FROM'])
df['NODE_NAME_TO'] = le.fit_transform(df['NODE_NAME_TO'])
df['PKT_CLASS'] = le.fit_transform(df['PKT_CLASS'])
# list_encode = le.inverse_transform(df['PKT_CLASS'])
# print(list_encode)

# config param for model
model_name = 'rf_clf_1.sav'
n_trees = 50

# split data
X = df.iloc[:,:27]
y = df.iloc[:,27]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

rf  = RandomForestClassifier(n_estimators = n_trees)
rf.fit(X_train, y_train)

# save model
pickle.dump(rf, open(model_name, 'wb'))

# load model
model_clf = pickle.load(open(model_name, 'rb'))

# show result
predict_train = model_clf.predict(X_train)
predict_test = model_clf.predict(X_test)

print('Accuracy of RF classifier on training set: {:.2f}'.format(model_clf.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'.format(model_clf.score(X_test, y_test)))
print('confusion matrix:')
print(confusion_matrix(y_test,predict_test))