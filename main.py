from fastapi import FastAPI
from pydantic import BaseModel
import pickle


PENGUIN_CLASS = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

with open("./src/model2.pkl", "rb") as file:
    load_model= pickle.load(file)

class Penguin(BaseModel):
    cl: float
    cd: float
    fl: float


class PenguinPrediction(BaseModel):
    features: Penguin
    prediction: str


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/pred", response_model=Penguin)
def get_prediction(cl: float, cd: float, fl:float):
    pred = load_model.predict([[cl, cd, fl]])[0]
    print(pred)
    pred = PENGUIN_CLASS[pred]

    return {
        "features": {
            "cl": cl,
            "cd": cd,
            "fl": fl,
        },
        "prediction": pred
    }




if __name__ == "__main__":
    import uvicorn
    print(__name__)
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)