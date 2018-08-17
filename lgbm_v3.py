import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
training_set=pd.read_csv('~/hcdr/application_train.csv')
test_set=pd.read_csv('~/hcdr/application_test.csv')
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

import lightgbm as LGB
import gc

def model(features, test_features, encoding='ohe', n_folds=5):
    #extract ids
    train_ids=features['SK_ID_CURR']
    test_ids=test_features['SK_ID_CURR']
    #extract labels for training
    labels=features['TARGET']
    #remove ids and TARGET
    features=features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features=test_features.drop(columns=['SK_ID_CURR'])
    # one hot encoding
    if encoding=='ohe':
        features=pd.get_dummies(features)
        test_features=pd.get_dummies(test_features)
        features, test_features=features.align(test_features, join='inner', axis=1)
        # no categorical indices to record
        cat_indices='auto'
    elif encoding=='le':
        label_encoder=LabelEncoder()
        # list for storing categorical cat_indices
        cat_indices=[]
        # iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype=='object':
                features[col]=label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col]=label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                #record the categorical cat_indices
                cat_indices.append(i)
    # raise value error if label encoding scheme is not valid
    else:
        raise ValueError("encoding must be either 'ohe' or 'le'")

    print('Training data shape:', features.shape)
    print('Testing data shape:', test_features.shape)

    # extract feature names
    feature_names=list(features.columns)
    # convert to numpy arrays
    features=np.array(features)
    test_features=np.array(test_features)
    # create the K fold object
    k_fold=KFold(n_splits=n_folds, shuffle=True, random_state=50)
    # empty array for feature importances
    feature_importance_values=np.zeros(len(feature_names))
    #empty array for test predictions
    test_predictions=np.zeros(test_features.shape[0])
    #empty array for out of fold validation predictions
    out_of_fold=np.zeros(features.shape[0])
    # lists for recording validation and training scores
    valid_scores=[]
    train_scores=[]
    #iterate through each k_fold
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels=features[train_indices], labels[train_indices]
        valid_features, valid_labels=features[valid_indices], labels[valid_indices]
        #create the model
        model=LGB.LGBMClassifier(nthread=4, n_estimators=10000, learning_rate=0.02,
        num_leaves=32, colsample_bytree=0.9497036, subsample=0.8715623,
        max_depth=8,reg_alpha=0.04,reg_lambda=0.073,min_split_gain=0.0222415,
        min_child_weight=40,silent=-1,verbose=-1)
        #train the model
        model.fit(train_features, train_labels, eval_metric='auc',
        eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
        eval_names=['valid','train'], categorical_feature=cat_indices,
        early_stopping_rounds=100, verbose=200)
        #record the best iteration
        best_iteration=model.best_iteration_
        #record the feature feature importances
        feature_importance_values+=model.feature_importances_/k_fold.n_splits
        #make predictions
        test_predictions+=model.predict_proba(test_features, num_iteration=best_iteration)[:,1]/k_fold.n_splits
        #record the out of fold predictions
        out_of_fold[valid_indices]=model.predict_proba(valid_features, num_iteration=best_iteration)[:,1]/k_fold.n_splits
        #record the best score
        valid_score=model.best_score_['valid']['auc']
        train_score=model.best_score_['train']['auc']
        valid_scores.append(valid_score)
        train_scores.append(train_score)

        # clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()

    #make the submission dataframe
    submission=pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    #make the feature importance DataFrame
    feature_importances=pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    #overall validation scores
    valid_auc=roc_auc_score(labels, out_of_fold)
    #add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    #needed for creating dataframe of validation scores
    fold_names=list(range(n_folds))
    fold_names.append('overall')
    #dataframe of validation scores
    metrics=pd.DataFrame({'fold': fold_names, 'train': train_scores, 'valid': valid_scores})

    return submission, feature_importances, metrics

submission, fi, metrics = model(training_set,test_set)
print('Baseline metrics')
print(metrics)

submission.to_csv('~/hcdr/lgbm_v3.csv', index=False)
