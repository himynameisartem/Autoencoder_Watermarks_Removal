import requests
import zipfile
import pickle

from app.config import (
    DATA_DIR,
    URL,
    ZIP_PATH,
    CACHE_DIR
)


def load_data():
    response = requests.get(URL, stream=True)
    response.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Download: {ZIP_PATH}")


def extract_data():
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"Extrct to: {DATA_DIR}")


def save_cache_data(data, name):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(CACHE_DIR / name, "wb") as f:
        pickle.dump(data, f)


def load_cache_data(name):
    with open(CACHE_DIR / name, "rb") as f:
        data = pickle.load(f)

    return data["train_no"], data["train_wm"], data["valid_no"], data["valid_wm"]
