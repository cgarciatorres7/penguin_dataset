from fastapi import FastAPI
from pydantic import BaseModel
import pickle


PENGUIN_CLASS = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

with open("Label_Encoder.pkl", "wb") as file:
    model= pickle.load(file)

class Penguin(BaseModel):
    culmen_lenght: float
    culmen_depht: float
    flipper_lenght: float

app = FastAPI()


@app.get("/", response_model=Penguin)
def get_prediction(cl: float, cd: float, fl:float):
    pred = model.predict([])
