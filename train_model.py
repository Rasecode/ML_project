print("1. Cargamos Librerias")
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler,PowerTransformer
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold, cross_val_predict, train_test_split,GridSearchCV,cross_val_score
import bentoml

params = {
   "boosting_type": "gbdt",
   "objective": "regression",
   "metric": {"a", "b"},
   "num_leaves": 31,
   "learning_rate": 0.05,
}

def train_model():
    print("2. Entrenando Modelo")
    df = pd.read_csv('data/heart.csv')
    numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns
    categorical = df.select_dtypes('object').columns
    y = df['HeartDisease']
    X= df.drop('HeartDisease', axis=1)
    y= df['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    ohe= OneHotEncoder()
    ct= make_column_transformer((ohe,categorical),remainder='passthrough')  
    model=LGBMClassifier(random_state=0)
    pipe = make_pipeline(ct, model)
    pipe.fit(X_train, y_train)

    return pipe

    # df = pd.read_csv('heart.csv')
    # numerical= df.drop(['HeartDisease'], axis=1).select_dtypes('number').columns
    # categorical = df.select_dtypes('object').columns
    # y = df['HeartDisease']
    # X= df.drop('HeartDisease', axis=1)
    # y= df['HeartDisease']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # ohe= OneHotEncoder()
    # ct= make_column_transformer((ohe,categorical),remainder='passthrough')  
    # model=LGBMClassifier(random_state=0)
    # # pipe = make_pipeline(ct, model)
    # # pipe.fit(X_train, y_train)
    # X_train=ct.fit_transform(X_train)
    # model.fit(X_train, y_train)
    # return model

if __name__ == "__main__":
    
    model=train_model()
    # saved_model=bentoml.lightgbm.save("my_beauty_model",model)
    saved_model=bentoml.sklearn.save_model("my_beauty_model",model)
    print(f"Modelo guardado: {saved_model}")