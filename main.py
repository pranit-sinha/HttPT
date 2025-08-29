import requests
from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import pipeline

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models['DistilBert'] = pipeline(model='distilbert-base-uncased-finetuned-sst-2-english')
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "App running"}

@app.get("/inference")
async def predict(input: str):
    return models['DistilBert'](input)[0]
