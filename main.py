from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pickle
import uvicorn
import logging as logger
from typing import Any

PENGUIN_CLASS = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

with open("./src/model2.pkl", "rb") as file:
    load_model = pickle.load(file)


class Penguin(BaseModel):
    cl: float
    cd: float
    fl: float


class PenguinPrediction(BaseModel):
    features: Penguin
    prediction: str


app = FastAPI()


@app.get("/")
async def root():
    return {"Hello": " World"}


@app.get("/pred", response_model=PenguinPrediction)
async def get_prediction(cl: float, cd: float, fl: float) -> Any:
    pred = load_model.predict([[cl, cd, fl]])[0]
    pred_class = PENGUIN_CLASS[pred]
    return {
        "features": {
            "cl": cl,
            "cd": cd,
            "fl": fl,
        },
        "prediction": pred_class
    }


@app.post("/json", response_model=PenguinPrediction)
async def get_prediction(penguin: Penguin) -> Any:
    cl = penguin.cl
    cd = penguin.cd
    fl = penguin.fl
    pred = load_model.predict([[cl, cd, fl]])[0]
    pred_class = PENGUIN_CLASS[pred]
    return {
        "features": {
            "cl": cl,
            "cd": cd,
            "fl": fl,
        },
        "prediction": pred_class
    }

@app.post("/csv", response_model=StreamingResponse)
async def get_predictions_from_csv(file: bytes = File(...)):
    file_content = file.decode("utf-8").split("\n")
    predictions = []
    for line in file_content:
        if line:
            cl, cd, fl = line.split(",")
            pred = load_model.predict([[cl, cd, fl]])[0]
            pred_class = PENGUIN_CLASS[pred]
            predictions.append({
                "features": {
                    "cl": cl,
                    "cd": cd,
                    "fl": fl,
                },
                "prediction": pred_class
            })


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8888, reload=True)
