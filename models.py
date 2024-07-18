# Basic Libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import mlflow
import gc
from warnings import filterwarnings
filterwarnings('ignore')

# Models Libs
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn import svm as svm
import lightgbm as lgbm
from lightgbm import LGBMRegressor

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

# Metric
from sklearn.metrics import r2_score

# Mlflow
import mlflow

# Features
import imp
import features as features
import config as config

def evaluate_model(model_name,X_train,X_test,y_train,y_test):
    # Linear Regression Model
    if model_name=='linear_regression':
        model =  GridSearchCV(
            estimator=LinearRegression(),
            param_grid=config.PARAMS_LINEAR_REGRESSION,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)
    
    # LASSO
    if model_name=='lasso':
        model =  GridSearchCV(
            estimator=Lasso(),
            param_grid=config.PARAMS_LASSO,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)

    # SVM Model
    if model_name=='svm':
        model =  GridSearchCV(
            estimator=svm.SVR(),
            param_grid=config.PARAMS_SVM,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)
        print(f'Finished training model {model_name}')

    # XGBoost
    if model_name=='xgb':
        model =  GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=config.PARAMS_XGB,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)
        print(f'Finished training model {model_name}')

    # LightGBM
    if model_name=='lgbm':
        model =  GridSearchCV(
            estimator=LGBMRegressor(),
            param_grid=config.PARAMS_DEFAULT,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)
        print(f'Finished training model {model_name}')

    # CatBoost
    if model_name=='catboost':
        model =  GridSearchCV(
            estimator=CatBoostRegressor(),
            param_grid=config.PARAMS_DEFAULT,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model.fit(X_train,y_train)
        # Predict on Test
        y_pred = model.predict(X_test)
        # Score 
        score = r2_score(y_test, y_pred)
        print(f'Finished training model {model_name}')
    
    # Ensemble
    if model_name=='ensemble':
        model_cat =  GridSearchCV(
            estimator=CatBoostRegressor(),
            param_grid=config.PARAMS_DEFAULT,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model_cat.fit(X_train,y_train)
        # Predict on Test
        y_pred_cat = model_cat.predict(X_test)

        model_xgb =  GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=config.PARAMS_XGB,
            cv=config.CV,
            n_jobs=-1,
            verbose=2,
            scoring='neg_mean_squared_error')
        # Train Model
        model_xgb.fit(X_train,y_train)
        # Predict on Test
        y_pred_xgb = model_xgb.predict(X_test)

        y_pred_final = (y_pred_xgb + y_pred_cat) / 2

        model = model_xgb
        # Score 
        score = r2_score(y_test, y_pred_final)
        print(f'Finished training model {model_name}')
    
    # 
    # Return
    return model_name,score,model.best_estimator_,model.best_params_

def run_experiment(model_name,exp_desc,X_train,X_test,y_train,y_test,valid_X_scaled):
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment(model_name)
    with mlflow.start_run():
        print('Model Name : ',model_name)
        model_name,score,best_model,best_param = evaluate_model(model_name,X_train,X_test,y_train,y_test)
        mlflow.log_param('drop_columns', config.DROP_COLUMNS)
        mlflow.log_param('model_name',model_name)
        mlflow.log_param('desc',exp_desc)
        mlflow.log_params(best_param)
        mlflow.log_param('cv',config.CV)
        mlflow.log_param('random_state',config.RANDOM_STATE)
        mlflow.log_param('features', str(list(valid_X_scaled.columns)))
        mlflow.log_metric('r2',score)
        mlflow.sklearn.log_model(best_model,model_name)
        # mlflow.log_artifact('transformed_data.csv')
    return best_model

def get_submission_csv(model,data,raw_valid):
    predictions = model.predict(data)
    sub_df = raw_valid[['id']]
    sub_df['FloodProbability'] = predictions
    sub_df.to_csv('/teamspace/studios/this_studio/2024/07/flood_prediction_notebook/data/submission.csv',index=False)
