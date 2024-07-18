import config as config
import data_handling as data_handler
import models as models
import features as features

if __name__=='__main__':
    raw_train, raw_valid = data_handler.load_data()

    X_train, X_test, y_train, y_test,valid_X,valid_X_scaled = data_handler.get_splits(raw_train,raw_valid,7)

    Experiment_Desc = 'Feature Func : 7 , CV = 6'
    # md_lr = run_experiment('linear_regression',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    # md_lgbm =run_experiment('lgbm',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    md_xgb = models.run_experiment('xgb',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)
    # md_catboost =models.run_experiment('catboost',Experiment_Desc,X_train,X_test,y_train,y_test,valid_X_scaled)

    models.get_submission_csv(md_xgb,raw_valid)