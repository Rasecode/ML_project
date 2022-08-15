print("1. Loading libraries")

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
import bentoml
import pickle 

params = {
   "boosting_type": "gbdt",
   "objective": "regression",
   "metric": {"a", "b"},
   "num_leaves": 31,
   "learning_rate": 0.05,
}

def data_processing():
    print("2. Train Test Split")
    df = pd.read_csv('data/heart.csv')
    numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns
    categorical = df.select_dtypes('object').columns
    y = df['HeartDisease']
    X= df.drop('HeartDisease', axis=1)
    y= df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_train.to_csv('data/heart_train_y.csv')

    print("3. One Hot Encoder")
    ohe= OneHotEncoder()
    ct= make_column_transformer((ohe,categorical),remainder='passthrough')  
    pipe = make_pipeline(ct)
    pipe.fit_transform(X_train)
    X_train = pipe.transform(X_train)
    np.savetxt('data/heart_train_tf.csv', X_train, delimiter=',')
    pickle.dump(pipe, open("processors/encoder.pkl", "wb"))

    print("3. Pre Processing Finished")

if __name__ == "__main__":
    
    data_processing()