import sqlite3
import pickle
from config import IMAGE_DATA_OUTPUT_PATH, DATABASE_PATH
import os


def create_table(conn):
    """Create a table in the database"""
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS images
                 (image_id INTEGER PRIMARY KEY, root TEXT, file TEXT)''')
    conn.commit()

def insert_data(conn, image_data):
    """Insert data into the database"""
    c = conn.cursor()
    rows = [(d["image_id"], d["root"], d["file"]) for d in image_data]
    c.executemany('INSERT INTO images VALUES (?,?,?)', rows)
    conn.commit()

def load_data(metadata_path):
    """Load the image data from the metadata file"""
    with open(metadata_path, "rb") as f:
        data= pickle.load(f)
    return data

def get_count(conn):
    """Get the number of rows in the database"""
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM images')
    count = c.fetchone()[0]
    conn.close()
    return count

def select_image_from_database(image_id, c):
    """Select an image from the database"""
    c.execute('SELECT root, file FROM images WHERE image_id = ?', (image_id,))
    result = c.fetchone()
    if result:
        root,file = result
        return os.path.join(root, file)

def generate_insert_into_database(database_path, metadata_path):
    """Generate the database and insert data into it"""
    conn = sqlite3.connect(database_path)
    create_table(conn)
    image_data = load_data(metadata_path)
    insert_data(conn, image_data)
    row_count = get_count(conn)
    print(f"Data inserted successfully. Row count: {row_count}")
    conn.close()

if __name__ == "__main__":
    generate_insert_into_database(DATABASE_PATH, IMAGE_DATA_OUTPUT_PATH)