"""
Download UNSW-NB15 dataset from Kaggle.

Requirements:
    pip install kaggle
    Set up ~/.kaggle/kaggle.json with your API credentials.
    See: https://github.com/Kaggle/kaggle-api#api-credentials

Usage:
    python data/download_data.py
"""

import os
import sys
import zipfile

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def download():
    try:
        import kaggle
    except ImportError:
        print("kaggle package not found. Install it with:")
        print("    pip install kaggle")
        sys.exit(1)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("Kaggle API credentials not found.")
        print("1. Go to https://www.kaggle.com/settings → API → Create New Token")
        print(f"2. Place the downloaded kaggle.json at: {kaggle_json}")
        sys.exit(1)

    print("Downloading UNSW-NB15 dataset from Kaggle …")
    os.system(f'kaggle datasets download mrwellsdavid/unsw-nb15 -p "{DATA_DIR}"')

    zip_path = os.path.join(DATA_DIR, 'unsw-nb15.zip')
    if os.path.exists(zip_path):
        print("Extracting …")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print(f"Dataset extracted to {DATA_DIR}/")
    else:
        print("Download may have failed. Check your Kaggle credentials and internet connection.")


if __name__ == '__main__':
    download()
