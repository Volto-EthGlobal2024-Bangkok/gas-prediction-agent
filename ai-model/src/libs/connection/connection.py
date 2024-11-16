from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URI = "mongodb://localhost:27017"  # Change this to your MongoDB URI
DATABASE_NAME = "productiondb"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]