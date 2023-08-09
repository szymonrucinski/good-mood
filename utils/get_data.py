"""Download EMO-DB dataset and unzip it in the data/raw folder."""
import logging
import os
from pathlib import Path

import coloredlogs

coloredlogs.install()
logger = logging.getLogger(__name__)

# if data raw does not exist create it
data_dir = Path(Path(__file__).parents[1], "data/raw/")

if not os.path.exists(data_dir):
    logger.info("Creating data/raw folder")
    os.mkdir(data_dir)

logging.info("Downloading EMO-DB dataset")
os.system("curl http://emodb.bilderbar.info/download/download.zip -O")
os.system(f"unzip download.zip -d {data_dir} && rm download.zip")
logging.info("Download completed!")
