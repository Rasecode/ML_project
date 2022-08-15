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

def train_model():
    print("2. Training Model")

    X_train = np.loadtxt('data/heart_train_tf.csv', delimiter=',')
    y_train = pd.read_csv('data/heart_train_y.csv')["HeartDisease"]
    model=LGBMClassifier(random_state=0)
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    return pipe

if __name__ == "__main__":
    
    model=train_model()
    saved_model=bentoml.sklearn.save_model("my_beautiful_model",model)
    print(f"3. Saved model: {saved_model}")