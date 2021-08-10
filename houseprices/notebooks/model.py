import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
# import xgboost import XGBRegressor

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

import pickle
import joblib

# Load data
train_X = pd.read_csv('../data/processed/train_X.csv')
train_y = pd.read_csv('../data/processed/train_y.csv')
test_X = pd.read_csv('../data/processed/test_X.csv')



def load_data():


def load_model():

def preprocessing():

...

def save_results():



# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(svr, gbr, rf),
                                meta_regressor=gbr,
                                use_features_in_secondary=True,
                                random_state=42)

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=train_X):
    rmse = np.sqrt(-cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


# Model Fitting
print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(train_X), np.array(train_y))

# print('xgboost')
# xgb_model_full_data = xgboost.fit(train_X,train_y)

print('svr')
svr_model_full_data = svr.fit(train_X, train_y)

print('RandomForest')
rf_model_full_data = rf.fit(train_X, train_y)

print('GradientBoosting')
gbr_model_full_data = gbr.fit(train_X, train_y)

# determine the threshold for missing values
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(test_X)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]

def handle_missing(features):
    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

test_X = handle_missing(test_X)

# Make prediction

prediction = stack_gen.predict(np.array(test_X))

# Save prediction as csv file 
numpy.savetxt("../data/output/prediction.csv", prediction, delimiter=",")

if name == "__main__":

    data = input()

    # loaded_data = load_data(data)
    # model = load_model()
    # processed_data = preprocessing(loaded_data)
    # preds = make_predictions(model, processed_data)]
    # save_results(preds)
    
    save_data()
