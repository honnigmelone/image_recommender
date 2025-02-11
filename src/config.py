import os

PATH_TO_IMAGE_DATA = r"D:\data\image_data"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMAGE_DATA_OUTPUT_PATH = os.path.join(DATA_ROOT, "image_data.pkl")
FEATURE_DATA_OUTPUT_PATH = os.path.join(DATA_ROOT, "feature_data.pkl")
DATABASE_PATH = os.path.join(DATA_ROOT, "image_database.db")
LOGS_ROOT = os.path.join(PROJECT_ROOT, "logs")