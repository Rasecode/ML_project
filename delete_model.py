print("1. Cargamos Librerias")
import bentoml

def model():
    models = bentoml.models.list()
    bentoml.models.delete("my_beauty_model:5s37hbazfo6mavhb")
    return models

if __name__ == "__main__":
    
    models=model()
    print(models)
    print("run successful")