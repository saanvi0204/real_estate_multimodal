"""
Satellite Image Fetcher
-----------------------
Fetches satellite images for properties using latitude and longitude
and saves them as PNG files.

Requirements:
- data/processed/train_clean.csv
- Columns: id, lat, long
- GOOGLE_MAPS_API_KEY stored in .env

Output:
- data/images/{property_id}.png
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

DATA_PATH = "../data/processed/test_clean.csv"
IMAGE_DIR = "../data/images/test"

MAP_TYPE = "satellite"
ZOOM = 19
IMAGE_SIZE = "256x256"
SCALE = 1

SLEEP_TIME = 0.15

load_dotenv()
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if API_KEY is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables.")

def build_image_url(lat: float, lon: float) -> str:
    """
    Construct Google Maps Static API URL.
    """
    return (
        "https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}"
        f"&zoom={ZOOM}"
        f"&size={IMAGE_SIZE}"
        f"&maptype={MAP_TYPE}"
        f"&scale={SCALE}"
        f"&key={API_KEY}"
    )

def save_image(image_bytes: bytes, filepath: Path):
    """
    Save image bytes to disk.
    """
    with open(filepath, "wb") as f:
        f.write(image_bytes)

def fetch_satellite_images():
    """
    Fetch satellite images for all properties in the dataset.
    """
    Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required_cols = {"id", "lat", "long"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    total = len(df)
    downloaded = 0
    skipped = 0
    failed = 0

    print(f"Starting satellite image download for {total} properties...\n")

    for _, row in tqdm(df.iterrows(), total=total):
        property_id = row["id"]
        lat = row["lat"]
        lon = row["long"]

        image_path = Path(IMAGE_DIR) / f"{property_id}.png"

        if image_path.exists():
            skipped += 1
            continue

        url = build_image_url(lat, lon)

        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                save_image(response.content, image_path)
                downloaded += 1
            else:
                failed += 1
                print(
                    f"[FAILED] ID {property_id} | HTTP {response.status_code}"
                )

        except Exception as e:
            failed += 1
            print(f"[ERROR] ID {property_id} | {str(e)}")

        time.sleep(SLEEP_TIME)

    print("\nDownload Summary")
    print("----------------")
    print(f"Total properties : {total}")
    print(f"Downloaded       : {downloaded}")
    print(f"Skipped (exists) : {skipped}")
    print(f"Failed           : {failed}")
    print("\nSatellite image fetching completed.")

if __name__ == "__main__":
    fetch_satellite_images()