from fastapi import FastAPI
from .pydantic_models import Observation, Prediction
import importlib
import pickle
from sklearn.linear_model import LogisticRegression

def load_model(model_name: str) -> LogisticRegression:
    with importlib.resources.open_binary("app.models", model_name) as f:
        model = pickle.load(f)
    return model

MODEL_NAME = "iris_regression.pickle"
model = load_model(MODEL_NAME)
app = FastAPI()

@app.get("/")
def status():
    # Check that the API is working.
    return "the API is up and running!"

CLASS_FLOWER_MAPPING = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica',
}

@app.post("/predict", status_code=201) # 201 is a successful POST request
def predict(obs: Observation) -> Prediction:
# For now, just return a dummy prediction
    # Return Prediction(flower_type="setosa")
    # .predict() gives us an array, but it has only one element
    prediction = model.predict(obs.as_dataframe())[0]
    flower_type = CLASS_FLOWER_MAPPING[prediction]
    pred = Prediction(flower_type=flower_type)
    return pred
