import requests
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
import base64
from io import BytesIO
from PIL import Image
from transformers import pipeline
from functools import lru_cache
from circuitbreaker import circuit
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='main.log', level=logging.INFO)

class ModelRegistry:
    def __init__(self):
        self.backends = {}
        self.config = {
                "sentiment-analysis": {
                    "name": "distilbert-base-uncased-finetuned-sst-2-english",
                    "task": "sentiment-analysis"
                    },
                "image-classification": {
                    "name": "google/vit-base-patch16-224",
                    "task": "image-classification"
                    }
                }

    def bootstrap(self):
        for service, config in self.config.items():
            try:                                                                                                                                                        
                self.backends[service] = pipeline(task=config["task"], model=config["name"], device=0)
            except Exception:
                raise RuntimeError("Failed to load ", service)

    def shutdown(self):
        self.backends.clear()

models = ModelRegistry()

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.bootstrap()
    yield
    models.shutdown()

app = FastAPI(version="0.2.2", lifespan=lifespan)

class InferenceRequest(BaseModel):
    input: str
    datatype: str

    @field_validator('datatype', mode='after')

    @classmethod
    def validate(cls, v):
        if v not in ['text', 'image']:
            raise ValueError('Only text and image supported currently.')
        return v

class InferenceResponse(BaseModel):
    service: str
    preds: list

@app.get("/")
async def root():
    return {"message": "App running"}

@lru_cache
@circuit
@app.post("/inference/{service}", response_model=InferenceResponse)
async def predict(service: str, request: InferenceRequest):
    model = models.backends.get(service)
    try:
        if request.datatype == 'image':
            logger.info('vit-base-patch16-224 called')
            raw = base64.b64decode(request.input)
            img = Image.open(BytesIO(raw))
            result = model(img)
        else:
            logger.info('distilbert-base-uncased-finetuned-sst-2-english called')
            result = model(request.input)
    except Exception:
        raise HTTPException(status_code=500, detail="Inference failed.")

    logger.info(f'Cache hits: {predict.cache_info().hits}, misses: {predict.cache_info().misses}')

    return InferenceResponse(service=service, preds=result if isinstance(result, list) else [result])

