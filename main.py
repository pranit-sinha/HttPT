import requests
import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple
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
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(filename='main.log', level=logging.INFO)

class BatchProcessor:
    def __init__(self, batchsize: int = 16, wait: float = 0.1):
        self.batchsize = batchsize
        self.wait = wait 
        self.queue = asyncio.Queue()
        self.results = {}
        self.processing = False
        
    async def get_req(self, req_id: str, service: str, input: str, datatype: str) -> dict:
        future = asyncio.get_running_loop().create_future()
        self.results[req_id] = future
        await self.queue.put((req_id, service, input, datatype, future))
        return await future
    
    async def fill_batch(self, models):
        while True:
            batch = []
            start_time = datetime.now()
            
            try:
                first_item = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                batch.append(first_item)
            except asyncio.TimeoutError:
                continue
                
            while len(batch) < self.batchsize:
                try:
                    timeout = self.wait - (datetime.now() - start_time).total_seconds()
                    if timeout <= 0:
                        break
                    item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                await self.process_batch(batch, models)
    
    async def process_batch(self, batch: List, models):
        group_by_service = {}
        for req_id, service, input, datatype, future in batch:
            if service not in group_by_service:
                group_by_service[service] = {"texts": [], "images": [], "futures": [], "req_ids": []}
            
            if datatype == "text":
                group_by_service[service]["texts"].append(input)
                group_by_service[service]["futures"].append(future)
                group_by_service[service]["req_ids"].append(req_id)
            else:  
                group_by_service[service]["images"].append(input)
                group_by_service[service]["futures"].append(future)
                group_by_service[service]["req_ids"].append(req_id)
        
        for service, data in group_by_service.items():
            if data["texts"]:
                model = models.backends[service]
                try:
                    results = model(data["texts"])
                    for future, result in zip(data["futures"], results):
                        future.set_result(result)
                except Exception as e:
                    for future in data["futures"]:
                        future.set_exception(e)
            
            for i, img_data in enumerate(data["images"]):
                try:
                    result = models.predict(service, img_data, "image")
                    data["futures"][i].set_result(result)
                except Exception as e:
                    data["futures"][i].set_exception(e)

class ModelRegistry:
    def __init__(self):
        self.backends = {}
        self.cache = TTLCache(maxsize=100, ttl=300)
        self.cache_info = {"hits": 0, "misses": 0}
        self.batch_processor = BatchProcessor()
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
        loop = asyncio.get_event_loop()
        self.batch_task = loop.create_task(self.batch_processor.fill_batch(self))

    def shutdown(self):
        if hasattr(self, 'batch_task'):
            self.batch_task.cancel()
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

@app.post("/inference/{service}/batch", response_model=InferenceResponse)
async def batch_predict(service: str, request: InferenceRequest):
    if service not in models.backends:
        raise HTTPException(status_code=404, detail="Service not found.")
    request_id = str(uuid.uuid4())
    try:
        result = await models.batch_processor.get_req(request_id, service, request.input, request.datatype)
    except Exception as e:
        raise HTTPException(status_code=500, detail="inference failed.")

    return InferenceResponse(service=service, preds=result if isinstance(result, list) else [result])

