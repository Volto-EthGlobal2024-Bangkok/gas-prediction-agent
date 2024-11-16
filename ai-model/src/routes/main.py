from fastapi import FastAPI, APIRouter
import uvicorn
import sys
sys.path.append('./src/models')
from inference import inference
sys.path.append('./src/libs')
from connection.connection import db
from gaspricedb.main import keep_running
import asyncio
from fastapi.middleware.cors import CORSMiddleware
sys.path.append('./src/scraper')
from scraper import fetch_last_week
from predict import load_model, predict
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow cookies
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/health")
async def root():
    return {"message": "Hello World"}

root_router = APIRouter(prefix="/api/v1", tags=["predict"])

@root_router.get("/predict/{days}", tags=["predict"])
async def read_users(days: int):
    days = int(days)

    if days > 5:
        raise HTTPException(status_code=404, detail="Too many days")

    prediction = await predict(model)
    print(prediction)
    return {"prediction": min(prediction.tolist())}

@root_router.get("/startScraper", tags=["scraper"])
async def start_scraper():
    fetch_last_week()
    return {"message": "Scraper started"}
app.include_router(root_router)

model = load_model()
# predictions = predict(model)
# print(predictions)

# asyncio.run(keep_running())
