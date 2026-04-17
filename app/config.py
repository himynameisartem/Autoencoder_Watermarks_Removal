from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ZIP_PATH = DATA_DIR / "watermarked.zip"
CACHE_DIR = DATA_DIR / "cache"

IMG_SIZE = 148
BATCH_SIZE = 32
SEED = 42
WM_TEXT = 'hi there'

URL = "https://storage.yandexcloud.net/academy.ai/watermarked.zip"
