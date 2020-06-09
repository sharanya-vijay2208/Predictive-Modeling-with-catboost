## Predictive-Modeling-with-catboost

This was my submission for the Kaggle competition Titanic: Machine Learning from Disaster https://www.kaggle.com/c/titanic
using Catboost to predict which passengers survived the Titanic shipwreck and it got me in the top 6% in the leaderboard.

CatBoost ,an open source gradient boosting machine learning library from Yandex used to solve both classification and regression tasks 
allows you to deal with categorical data and build models without having to encode it.
Usual gradient boosting algorithms have no support/no optimal support for categorical data which arises in many datasets.
This was my motivation to try it on the tabular heterogeneous Titanic data for classification.
Yandex claims that it provides great results without parameter tuning and that has been the case here(I tried
Bayesian optimization using hyperopt package).


## Prerequisites

Load train and test datasets to data folder

## Usage

python main.py 
