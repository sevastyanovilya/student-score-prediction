import os
import logging
from datetime import datetime

# создаём папку для логов, если её нет
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    format="%(asctime)s | %(levelname)s | %(lineno)d | %(message)s",
    level=logging.INFO,
)
