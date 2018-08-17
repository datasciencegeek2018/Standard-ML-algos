# Standard-ML-algos
Various algos were tested on data from a Kaggle competition HCDR (https://www.kaggle.com/c/home-credit-default-risk)

Since our objective was mainly to test the efficacy of various algorithms out of the box, we only leveraged the base dataset and not any of the incremental datasets made available to the competition

The evaluation score was AUC

The various algorithms performed very differently and are shown below along with their AUCs and corresponding file names

Algorithm | AUC | code file name
--------- | --- | --------------
Decision Trees | 0.525 | 
K Nearest Neighbors | 0.546 | 
Naive Bayes | 0.569 | 
Support Vector Machines | 0.596 | 
Logistic Regression with PCA | 0.655 | 
Random Forests | 0.708 | 
Logistic Regression with LDA | 0.737 | 
Logistic Regression C optimized | 0.739 | 
Catboost | 0.743 | 
LGBM | 0.75 | 

Logistic regression does pretty well. However, Gradient Boosting ensemble techniques like CatBoost / Light GBM are even better and achieve pretty good performance
