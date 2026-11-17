import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd

from fraud_detection.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from fraud_detection.exception.exception import FraudDetectionException 

app = FastAPI(title="Fraud Detection System", description="MLOps Production-Grade Fraud Detection API")

class TransactionData(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.get("/", tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict", tags=["Prediction"])
async def predict_fraud(data: TransactionData):
    try:
        custom_data = CustomData(
            step=data.step,
            type=data.type,
            amount=data.amount,
            oldbalanceOrg=data.oldbalanceOrg,
            newbalanceOrig=data.newbalanceOrig,
            oldbalanceDest=data.oldbalanceDest,
            newbalanceDest=data.newbalanceDest
        )

        final_df = custom_data.get_data_as_dataframe()

        prediction_pipeline = PredictionPipeline()
        prediction, probability = prediction_pipeline.predict(final_df)

        result = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
        score = float(probability[0][1]) if probability is not None else 0.0
        
        return {
            "prediction": result,
            "fraud_probability": round(score, 4)
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)