import schedule
import time
from datetime import datetime

import sys
sys.path.append('./src/libs')
from connection.connection import db
from web3 import Web3

INFURA_URL = "https://mainnet.infura.io/v3/" + INFURA_KEY  # Replace with your Infura project ID
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def fetch_and_store_latest_block():
    try:
        latest_block = web3.eth.get_block("latest")
        if "baseFeePerGas" in latest_block and latest_block["baseFeePerGas"] is not None:
            # Extract data
            timestamp = datetime.utcfromtimestamp(latest_block["timestamp"]).isoformat()
            gas_price_gwei = int(latest_block["baseFeePerGas"]) / 1e9

            # Prepare the document
            document = {
                "block_number": latest_block["number"],
                "timestamp": timestamp,
                "gas_price_gwei": round(gas_price_gwei, 9),
            }

            # Insert into MongoDB
            db.gas_price.insert_one(document)
            print(f"Inserted latest block: {latest_block['number']} with gas price {gas_price_gwei:.9f} Gwei")
        else:
            print("Base fee per gas is unavailable in the latest block.")

    except Exception as e:
        print(f"Error fetching the latest block: {e}")

# Schedule the job
schedule.every(24).minutes.do(fetch_and_store_latest_block)

# Keep the script running
async def keep_running():
    print("Starting scheduled task...")
    while True:
        schedule.run_pending()
        time.sleep(24*60)
