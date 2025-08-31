import requests
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, field_validator
import base64
from io import BytesIO
from PIL import Image
from transformers import pipeline
from cachetools import TTLCache
import hashlib 
import json
from circuitbreaker import circuit
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='main.log', level=logging.INFO)

class ModelRegistry:
    def __init__(self):
        self.backends = {}
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.cache_info = {"hits": 0, "misses": 0}
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
        logger.info(f'Cache hits: {self.cache_info["hits"]}, misses: {self.cache_info["misses"]}')    
        self.backends.clear()

    @circuit
    def predict(self, service: str, input: str, datatype: str):
        key = hashlib.md5(f'{service}{input}'.encode()).hexdigest()
        if key in self.cache:
            self.cache_info["hits"] += 1
            return self.cache[key]
        else:
            self.cache_info["misses"] += 1

        model = self.backends.get(service)
        if datatype == 'image':
            raw = base64.b64decode(input)
            img = Image.open(BytesIO(raw))
            result = model(img)
            logger.info('vit-base-patch16-224 called')
        else:
            result = model(input)
            logger.info('distilbert-base-uncased-finetuned-sst-2-english called')

        self.cache[key] = result
        return result

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

@app.post("/inference/{service}", response_model=InferenceResponse)
async def predict(service: str, request: InferenceRequest):
    if service not in models.backends:
        raise HTTPException(status_code=404, detail="Service not found.")
    try:
        result = models.predict(service, request.input, request.datatype)
    except Exception:
        raise HTTPException(status_code=500, detail="inference failed.")

    return InferenceResponse(service=service, preds=result if isinstance(result, list) else [result])

