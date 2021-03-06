import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
training_set=pd.read_csv('~/hcdr/application_train.csv')
test_set=pd.read_csv('~/hcdr/application_test.csv')
from sklearn.preprocessing import OneHotEncoder

le=LabelEncoder()
le_count=0
for col in training_set:
    if training_set[col].dtype=='object':
        if len(list(training_set[col].unique()))<=2:
            le.fit(training_set[col])
            print(f'this {col} has {le.classes_}')
            training_set[col]=le.transform(training_set[col])
            test_set[col]=le.transform(test_set[col])
            le_count+=1
print(f'{le_count} columns were label encoded')

training_set=pd.get_dummies(training_set)
test_set=pd.get_dummies(test_set)
training_labels=training_set['TARGET']
training_set, test_set=training_set.align(test_set, join='inner', axis=1)
print(f'training features shape: {training_set.shape}')
print(f'test features shape: {test_set.shape}')
training_set['TARGET']=training_labels

from sklearn.preprocessing import MinMaxScaler,Imputer
X_train=training_set.drop(columns=['TARGET'])
X_test=test_set.copy()
features=list(X_train.columns)
imputer=Imputer(strategy='median')
scaler=MinMaxScaler(feature_range=(0,1))
imputer.fit(X_train)
X_train=imputer.transform(X_train)
X_test=imputer.transform(X_test)
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(f'training data shape:{X_train.shape}')
print(f'test data shape:{X_test.shape}')

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2, n_jobs=-1)
classifier.fit(X_train, training_labels)
classifier_pred=classifier.predict_proba(X_test)[:,1]
submit=test_set[['SK_ID_CURR']]
submit['TARGET']=classifier_pred
submit.to_csv('~/hcdr/knn_baseline_k5.csv', index=False)
