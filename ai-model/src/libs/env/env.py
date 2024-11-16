from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get environment variables
INFURA_KEY = os.getenv("INFURA_KEY")