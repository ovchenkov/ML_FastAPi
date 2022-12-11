from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

app = FastAPI()

model, coef, intercept, enc, scaler = pickle.load(
    open("model.pkl", 'rb'))


class Item(BaseModel):
    year: int
    km_driven: int
    mileage: float
    engine: int
    max_power: int
    power_by_volume: float
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    seats: str


@app.post('/predict_item')
async def predict_item(item: Item):
    X_cat = pd.DataFrame({'fuel': [item.fuel],
                          'seller_type': [item.seller_type],
                          'transmission': [item.transmission],
                          'owner': [item.owner],
                          'seats': [item.seats]
                          })

    X_num = pd.DataFrame({'year': [item.year],
                          'km_driven': [item.km_driven],
                          'mileage': [item.mileage],
                          'engine': [item.engine],
                          'max_power': [item.max_power],
                          'power_by_volume': [item.power_by_volume]
                          })

    enc = OneHotEncoder(drop="first")
    enc.fit(X_cat)
    X_cat = enc.transform(X_cat).toarray()
    X_cat = pd.DataFrame(X_cat)
    X = pd.concat([X_num, X_cat], axis=1, join='inner')
    prediction = model.predict(X)

    return prediction[0][0]
