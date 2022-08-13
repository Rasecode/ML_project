### CREATING NEW RUNNER
### RESOURCES:
# https://docs.bentoml.org/en/latest/concepts/runner.html

from typing import Any
import bentoml
from bentoml.io import JSON
from bentoml.io import NumpyNdarray, PandasDataFrame, JSON
import pandas as pd
import numpy as np
from pydantic import BaseModel

class Customer(BaseModel):
    Age: int = 63 
    Sex: str = "F"
    ChestPainType: str = 'ATA'
    RestingBP: int = 140
    Cholesterol: int = 195
    FastingBS: int = 0
    RestingECG: str = "Normal"
    MaxHR: int = 179
    ExerciseAngina: str = "N"
    Oldpeak: float = 0.0
    ST_Slope: str = "Up"

bento_model = bentoml.sklearn.get("my_beauty_model:latest")

def Extract(lst):
    return [item[-1] for item in lst][0]

class Heartpredict(bentoml.Runnable):
    # SUPPORTED_RESOURCES = ("cpu",)
    # SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        # load the model instance
        self.model = bentoml.sklearn.load_model("my_beauty_model:latest")

    @bentoml.Runnable.method(batchable=False)
    def predict_proba(self, input_data):
        return self.model.predict_proba(input_data)

heart_predict_runner = bentoml.Runner(Heartpredict, models=[bento_model])
service = bentoml.Service("heart_disease_predictor", runners=[heart_predict_runner])


@service.api(input=JSON(pydantic_model=Customer), output=NumpyNdarray())
def predict(data: Customer) -> np.ndarray:

    #customer into dataframe
    df = pd.DataFrame(data.dict(),index=[0])

    results = heart_predict_runner.predict_proba.run(df)
    return Extract(results)