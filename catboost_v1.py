import numpy as np
import pandas as pd

training_set=pd.read_csv('~/hcdr/application_train.csv')
test_set=pd.read_csv('~/hcdr/application_test.csv')

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import catboost as CB
import gc

#replacing missing categorical data with a new missing category and
#missing non categorical data with the median of specific variable in
#training data and replicating in test data eval_set

lookup={}
for col in training_set:
    if training_set[col].dtype=='object':
        training_set[col]=training_set[col].fillna('missing')
    else:
        lookup[col]=training_set[col].median()
        training_set[col]=training_set[col].fillna(training_set[col].median())

for col in test_set:
    if test_set[col].dtype=='object':
        test_set[col]=test_set[col].fillna('missing')
    else:
        test_set[col]=test_set[col].fillna(lookup[col])

def model(features, test_features, n_folds=5):
    #extract ids
    train_ids=features['SK_ID_CURR']
    test_ids=test_features['SK_ID_CURR']
    #extract labels for training
    labels=features['TARGET']
    #remove ids and TARGET
    features=features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features=test_features.drop(columns=['SK_ID_CURR'])
    # determine indices of categorical features
    features, test_features=features.align(test_features, join='inner', axis=1)
    cat_indices=[]
    for col in features:
        if features[col].dtype == 'object':
            cat_indices.append(features.columns.get_loc(col))

    print('Training data shape:', features.shape)
    print('Testing data shape:', test_features.shape)

    # extract feature names
    feature_names=list(features.columns)
    # convert to numpy arrays
    features=np.array(features)
    test_features=np.array(test_features)
    # create the K fold object
    k_fold=KFold(n_splits=n_folds, shuffle=True, random_state=50)
    #empty array for test predictions
    test_predictions=np.zeros(test_features.shape[0])
    #empty array for out of fold validation predictions
    out_of_fold=np.zeros(features.shape[0])
    # lists for recording validation and training scores
    valid_scores=[]
    #iterate through each k_fold
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels=features[train_indices], labels[train_indices]
        valid_features, valid_labels=features[valid_indices], labels[valid_indices]
        #create the model
        model=CB.CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss',
        bootstrap_type='Bernoulli', eval_metric='AUC', one_hot_max_size=255, class_weights=[1,2],
        od_type='Iter')

        #train the model
        model.fit(train_features, train_labels,
        eval_set=(valid_features, valid_labels),
        cat_features=cat_indices, use_best_model=True)
        #make predictions
        test_predictions+=model.predict_proba(test_features)[:,1]/k_fold.n_splits
        #record the out of fold predictions
        out_of_fold[valid_indices]=model.predict_proba(valid_features)[:,1]/k_fold.n_splits

        # clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    #make the submission dataframe
    submission=pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    #overall validation scores
    valid_auc=roc_auc_score(labels, out_of_fold)
    #add the overall scores to the metrics
    valid_scores.append(valid_auc)
    metrics=valid_scores

    return submission, metrics

submission, metrics = model(training_set,test_set)
print('Baseline metrics')
print(metrics)

submission.to_csv('~/hcdr/catboost_v1.csv', index=False)
