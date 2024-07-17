import features as features
def process_data(train_data,func_num,initial_features):
    fun_dict = {
        1: features.create_features_1,
        2: features.create_features_2,
        3: features.create_features_3,
        4: features.create_features_4,
        5: features.create_features_5,
        6: features.create_features_6
    }
    feature_func = fun_dict.get(func_num)
    df = train_data.copy()
    df = feature_func(df,initial_features)
    # Drop Columns
    # df.drop(columns=DROP_COLUMNS,axis=1,inplace=True)
    return df