import config as config
import data_handling as data_handler
import models as models
import features as features
import logging

PERFORM_PCA = True
FEATURE_FUNC = 9
PCA_N_COMP = 10

if __name__=='__main__':
    raw_train, raw_valid = data_handler.load_data()
    X_train, X_test, y_train, y_test,valid_X,valid_X_scaled = data_handler.get_splits(raw_train,raw_valid,FEATURE_FUNC)

    if PERFORM_PCA:
        X_train, X_test, y_train, y_test,valid_X,valid_X_scaled = data_handler.perform_PCA(X_train, X_test, y_train, y_test,valid_X,valid_X_scaled,PCA_N_COMP)

    Experiment_Desc = f"Feature Func : {FEATURE_FUNC}, CV = {config.CV}, RANDOM_STATE = {config.RANDOM_STATE}, PCA : {PERFORM_PCA}"
    md_lr = models.run_experiment('linear_regression',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    # md_lgbm = models.run_experiment('lgbm',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    # md_xgb = models.run_experiment('xgb',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    # md_catboost =models.run_experiment('catboost',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)

    which_model = md_lr
    # models.get_submission_csv(which_model,valid_X_scaled,raw_valid)