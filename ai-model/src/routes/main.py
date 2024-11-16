from fastapi import FastAPI, APIRouter
import uvicorn
import sys
sys.path.append('./src/models')
from inference import inference
sys.path.append('./src/libs')
from connection.connection import db
from gaspricedb.main import keep_running
import asyncio
sys.path.append('./src/scraper')
from scraper import fetch_last_week


app = FastAPI()

@app.get("/health")
async def root():
    return {"message": "Hello World"}

root_router = APIRouter(prefix="/api/v1", tags=["predict"])

@root_router.get("/predict/{days}", tags=["predict"])
async def read_users(days: int):
    days = int(days)
    print(days)
    if days > 5:
        raise HTTPException(status_code=404, detail="Too many days")

    return {"prediction": inference(days)}

@root_router.get("/startScraper", tags=["scraper"])
async def start_scraper():
    fetch_last_week()
    return {"message": "Scraper started"}
app.include_router(root_router)

# asyncio.run(keep_running())
