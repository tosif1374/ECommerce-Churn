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

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Divide by zero")
    print(f"Log file created at: {LOG_FILE_PATH}")
