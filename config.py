TEST_SIZE = 0.2
DROP_COLUMNS = ['id']
CV = 5
RANDOM_STATE = 10
ORIGINAL_COLS = ['id','MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors', 'FloodProbability']

INITIAL_FEATURES = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
       'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
       'Siltation', 'AgriculturalPractices', 'Encroachments',
       'IneffectiveDisasterPreparedness', 'DrainageSystems',
       'CoastalVulnerability', 'Landslides', 'Watersheds',
       'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
       'InadequatePlanning', 'PoliticalFactors']

# GridSearchCV Params
PARAMS_LINEAR_REGRESSION = {}

PARAMS_XGB = {
    'max_depth':[3,5,7],
    'booster':['gbtree','gblinear','dart'],
}

PARAMS_RANDOM_FOREST_REGRESSION = {
    'n_estimators':[100,200], 
    # 'criterion':['squared_error','absolute_error','friedman_mse','poisson']
}

PARAMS_LASSO = {
    'alpha':[0.1,0.01], 
    'max_iter':[100,500]
}

PARAMS_RIDGE = {
    'alpha':[0.1,0.01], 
    'max_iter':[100,500]
}

PARAMS_SVM = {
    'C': [1, 10], 
    'kernel': ['linear', 'rbf']
}

PARAMS_DEFAULT = {}