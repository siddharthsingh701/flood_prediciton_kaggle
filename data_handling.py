import features as features
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import config as config
from sklearn.decomposition import PCA

def load_data():
    raw_train = pd.read_csv('/teamspace/studios/this_studio/2024/07/flood_prediction_notebook/data/train.csv')
    raw_valid= pd.read_csv('/teamspace/studios/this_studio/2024/07/flood_prediction_notebook/data/train.csv')
    return raw_train,raw_valid

def process_data(train_data,func_num):
    fun_dict = {
        1: features.create_features_1,
        2: features.create_features_2,
        3: features.create_features_3,
        4: features.create_features_4,
        5: features.create_features_5,
        6: features.create_features_6,
        7: features.create_features_7,
        8: features.create_features_8,
        9: features.create_features_9
    }
    feature_func = fun_dict.get(func_num)
    df = train_data.copy()
    df = feature_func(df)
    # Drop Columns
    # df.drop(columns=DROP_COLUMNS,axis=1,inplace=True)
    return df

def get_splits(raw_train,raw_valid,feature_func):

    # Train Test Split and Scaling
    raw_train_X = raw_train.drop('FloodProbability',axis=1)
    raw_train_y = raw_train['FloodProbability']

    train_X = process_data(raw_train_X,feature_func)
    train_y = raw_train_y

    sc = StandardScaler()
    train_X_scaled = sc.fit_transform(train_X)
    train_X_scaled = pd.DataFrame(train_X_scaled,columns=train_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(train_X_scaled, train_y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    valid_X = process_data(raw_valid,feature_func)
    valid_X_scaled = sc.transform(valid_X)
    valid_X_scaled = pd.DataFrame(valid_X_scaled,columns=valid_X.columns)
    return X_train, X_test, y_train, y_test,valid_X,valid_X_scaled

def perform_PCA(X_train, X_test, y_train, y_test,valid_X,valid_X_scaled,n_components):
    # Perform PCA
    pca = PCA(n_components)  # Choose the number of components you want to keep
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca,X_test_pca,y_train,y_test,valid_X,valid_X_scaled