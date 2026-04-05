import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.cxgboost import CXGBoost

app = FastAPI(title="CausalCare API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = CXGBoost.load("model/c-xgboost")


class PatientProfile(BaseModel):
    age: float
    sex: float
    race_eth: float
    education_yrs: float
    poverty_cat: float
    employ_status: float
    region: float
    self_health: float
    heart_disease: float
    diabetes: float
    hypertension: float
    asthma: float
    cancer: float


@app.get("/")
def root():
    return {"status": "ok", "model": "C-XGBoost", "problem": "Insurance Coverage Effect"}


@app.post("/predict/coverage")
def predict_coverage(patient: PatientProfile):
    features = [list(patient.model_dump().values())]
    pred = model.predict(features)

    y0 = float(np.expm1(pred["y_0_hat"][0]))
    y1 = float(np.expm1(pred["y_1_hat"][0]))
    ps = float(pred["propensity_score"][0])
    cate = y1 - y0

    return {
        "cate":             round(cate, 2),
        "y0_hat":           round(y0, 2),
        "y1_hat":           round(y1, 2),
        "propensity_score": round(ps, 4),
        "interpretation":   (
            f"Gaining insurance coverage is estimated to "
            f"{'increase' if cate >= 0 else 'decrease'} this patient's "
            f"annual medical expenditure by ${abs(cate):,.0f}, "
            f"reflecting {'greater' if cate >= 0 else 'lower'} access to "
            f"and utilization of healthcare services."
        ),
    }
    
    
# import numpy as np
# from utils.cxgboost import CXGBoost

# model = CXGBoost.load('model/c-xgboost')

# input = {'age': 77,
#  'sex': 2,
#  'race_eth': 3,
#  'education_yrs': 6,
#  'poverty_cat': 3,
#  'employ_status': 3,
#  'region': 21,
#  'self_health': 3,
#  'heart_disease': 2,
#  'diabetes': 1,
#  'hypertension': 1,
#  'asthma': 2,
#  'cancer': 2}

# pred = model.predict([list(input.values())])

# print("[INFO] y0_hat:", np.expm1(pred['y_0_hat'][0]))
# print("[INFO] y1_hat:", np.expm1(pred['y_1_hat'][0]))
# print("Propensity score:", 100*pred['propensity_score'][0])