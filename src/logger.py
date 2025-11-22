# /mnt/data/logger.py
import logging
import os
from datetime import datetime
import sys

# Create logs directory if not exists
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create dynamic log filename based on date and time
LOG_FILE = datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure the root logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    filemode="a",
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Named logger (you can import either `logging` or `logger` from this module)
logger = logging.getLogger(__name__)

# Export `logging` name so `from src.logger import logging` keeps working
# (this is simply the configured logging module)
# Note: other modules can also do `from src.logger import logger`
if __name__ == "__main__":
    logger.info("Logger initialized")
    print(f"Log file created at: {LOG_FILE_PATH}")
