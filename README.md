# MLops_project
### In this project we will develop an API that will provide us the probability of suffering a heart disease based on a machine learning model.
### To start, https://www.kaggle.com/kaanboke/beginner-friendly-catboost-with-optuna, will be the resource for our machine learning model, specifically we will use the LightGBM section with its Encoder.
### The Repo has the next structure:
```
MLops_project
│
│  __init__.py
│  bentofile.yaml
│  README.md
│  requirements.txt
│
└───notebooks
│   │   ...
│
└───resources_url_etc
│   │   ...
│
└───src
│   │   __init__.py
│   │   bentoml_app.PY
│   │   predict.PY
│   │   process_data.PY
│   │
│   └───data
│   │   │   heart_train_tf.csv
│   │   │   heart_train_y.csv
│   │   │   heart.csv
│   │   
│   └───processors
│       │   encoder.pkl
│
└───streamlit
    │   app.py
    │   Dockerfile
    │   requirements.txt
```
Basically we will use BentoML, which is an open-source framework for serving ML models at production scale, Docker for the Containerization, Streamlit for the UI and last but not least GCP, specifically Cloud Run in order to push our images in a serverless way.

Step 1:
We have to install all the libraries, run the first requirements.txt (MLops_project/requirements.txt) using:
```
pip install -r MLops_project/requirements.txt
```
Step 2:
We have to create our model, so execute process_data.py and then predict.py
```
python process_data.py
python predict.py
```
Those commands will create first our encoder.pkl and our machine learning model which will be save using bentoml internally (you can check the .py archives for more details)

Step 3:
Once our model is build, we could try creating our first local API running the next command
```
bentoml serve src/bentoml_app.py:service --reload
```
We could check if the deployment was good opening http://localhost:3000/ and trying our API

Step 4:
When our model is working well, we are able to create our Bento, but first we have to create our bentofile.yaml file to specify the path of our service and the requirements for our project (for more detail, check the .yaml file). Try;
```
bentoml serve src/bentoml_app.py:service --reload
```
Step 5:
Once we check everything is going well, we can dockerize our model ir order to start our deployment in any Cloud. Try:
```
bentoml containerize heart_disease_predictor:""tag""
```
Step 6:
We can do a double check of our API app, running our container and open again http://localhost:3000/. I believe is a good practice make double check whenever is posible.
```
docker run -it --rm -p 3000:3000 heart_disease_predictor:""tag""
```
Step 7:
So now, we have our docker image stored locally, we have to push it to DockerHub in order to start our deployment in GCP Cloud Run:
```
#First create a new tag, in this case "rasecado" is my DockerHub usser, so change it to yours.
docker tag heart_disease_predictor:""tag"" rasecado/heart_disease_predictor:""tag""
docker push rasecado/heart_disease_predictor:""tag""
```
Step 8:
We need to go to GCP, search for Cloud Run and habilitate Registry Api and Cloud Run Api, in order to start our deployment.
Once is done, we have to pull first our DockerHub repo and then push to Registry.
```
docker pull rasecado/heart_disease_predictor:""tag""
docker tag rasecado/heart_disease_predictor:""tag"" gcr.io/rimac-test-359419/heart_disease_predictor:""tag""
docker push gcr.io/rimac-test-359419/heart_disease_predictor:""tag""
#Note:
#You should change the project name "rimac-test-359419" to yours and obviously select the tag that you want
```
Step 8:
So, the last step is just customize our Cloud Run service, and select our image to run in GCP.

BUILDING OUR UI:
In this part, we will deploy the UI for our machine learning model. Check streamlit folder for more detail.

Step 1:
Create a .py with all tne necessities for your UI. You can take streamlit/app.py as a template.
Create a Dockerfile, in order to conteinerize our app.py.  You can take streamlit/Dockerfile as a template.
Create a requirements.txt for our Dockerfile.

Step 2:
Run the following command to check if your UI is working well, once is done you can check it clicking the localhost Url provided by the command prompt. Is usually http://localhost:8501/
```
streamlit run app.py
```
Step 3:
Run the next commands to build your docker image. remember you must execute ir on the directory where the dockerfile is stored.
```
docker build -t streamlitapp:latest .
```
Step 4:
Lastly, you can make a double check of your UI app executing your docker image.
```
docker run -p 8501:8501 streamlitapp:latest
```
Step 5:
Follow the same bentoml model deployment in GCP steps in order to deploy your UI app in GCP Cloud Run.

-------------------------------------------------------------------------------------------------------------
Summary:

To sum up we combine different frameworks for creating our API. But the main ones are BentoML, Docker, Streamlit, and Google Cloud.

API Functionality:
-Once the containers are created, the streamlit container use the API container in order to produce the requests.
-In order to not touch the API itself I decided to create two containers instead of one, so for next updates in the UI, the redeployment will be easy.
Also, retraining the model will be simple if we follow the steps, due to is a simple model and it does not take too much time for preprocessing, considering that this is just a test, too.
-The time of response is fast, it does not take more than 5 seconds for the first request, and the next ones are faster, basically instantaneous.
-The use of the API is Simple, we have a very intuitive UI and it obviously does not take unvalid values for our model, so they are limited so some options in the case of categorical variables and limited max and min in the case of the continuous (for more detail check streamlit/app.py)

Notes:
Resources_url_etc folder will be organize in the next days, so wait for the last update, anyway, there are many links, articles and videos that I use for my deployment.

Finally here are the GCP deployed images links:

https://heart-disease-predictor-ioqjxmu3cq-uc.a.run.app
https://streamlitapprimac-ioqjxmu3cq-uc.a.run.app

