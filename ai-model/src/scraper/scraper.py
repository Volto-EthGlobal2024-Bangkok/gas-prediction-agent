import schedule
import time
from datetime import datetime

import sys
sys.path.append('./src/libs')
from connection.connection import db
from web3 import Web3

from env.env import INFURA_KEY

INFURA_URL = "https://mainnet.infura.io/v3/" + INFURA_KEY  # Replace with your Infura project ID
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def fetch_last_week():
    try:
        latest_block = web3.eth.get_block("latest")
        latest_block_number = latest_block["number"]

        start_block = latest_block_number - 120 * 3 * 24 * 7
        for block_number in range(start_block, latest_block_number, 120):
            block = web3.eth.get_block(block_number)
            if "baseFeePerGas" in block and block["baseFeePerGas"] is not None:
                # Extract data
                timestamp = datetime.utcfromtimestamp(block["timestamp"]).isoformat()
                gas_price_gwei = int(block["baseFeePerGas"]) / 1e9

                # Prepare the document
                document = {
                    "block_number": block["number"],
                    "timestamp": timestamp,
                    "gas_price_gwei": round(gas_price_gwei, 9),
                    "day": timestamp.split("T")[0],
                }

                # Insert into MongoDB
                db.gasprices.insert_one(document)
                print(f"Inserted block: {block['number']} with gas price {gas_price_gwei:.9f} Gwei")
            else:
                print(f"Base fee per gas is unavailable in block {block_number}")

    except Exception as e:
        print(f"Error fetching the latest block: {e}")

