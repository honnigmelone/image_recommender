import sqlite3
import pickle
from config import IMAGE_DATA_OUTPUT_PATH, DATABASE_PATH
import os


def create_table():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (image_id INTEGER PRIMARY KEY, root TEXT, file TEXT, size TEXT)''')
    conn.commit()
    conn.close()

def insert_data(image_data):
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    rows = [(d["image_id"], d["root"], d["file"], d["size"]) for d in image_data]
    c.executemany('INSERT INTO images VALUES (?,?,?,?)', rows)
    conn.commit()
    conn.close()

def load_data():
    with open(IMAGE_DATA_OUTPUT_PATH, "rb") as f:
        data= pickle.load(f)
    return data

def get_count():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM images')
    count = c.fetchone()[0]
    conn.close()
    return count

def main():
    create_table()
    image_data = load_data()
    insert_data(image_data)
    row_count = get_count()
    print(f"Data inserted successfully. Row count: {row_count}")

if __name__ == "__main__":
    main()