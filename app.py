from fastapi import FastAPI
from pydantic import BaseModel
from inference_onnx import IeltsONNXPredictor
app = FastAPI(title="IELTS Scoring App")

predictor = IeltsONNXPredictor()

@app.get("/")

class TextData(BaseModel):
    text: str
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def predict(question: str, essay: str):
    input = 'CLS '+ question + ' SEP ' + essay + ' SEP'
    predicted_task, predicted_coh = predictor.predict(essay, input)
    return {"predicted_task": predicted_task, "predicted_coherence": predicted_coh}