import sqlite3
import pytest
import os
import pickle
import sys 

# weil sich die database.py nicht im selber ordner befindet (sieht so cleaner aus :D)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from database import create_table, insert_data, load_data, get_count, select_image_from_database


# test datenbank aufrufen
@pytest.fixture
def db_connection():
    """
    erstellt eine temporäre SQLite-Datenbank im speicher für tests
    """
    # memory datenbank für unseren temporären test - also sozusagen keine echte datenbank
    conn = sqlite3.connect(":memory:") 

    # erstellt die tabelle selber
    create_table(conn)
    return conn

@pytest.fixture
def sample_image_data():
    """
    liefert beispiel-bildmetadaten für tests
    """
    return [
        {"image_id": 1, "root": "/test/path", "file": "image1.jpg"},
        {"image_id": 2, "root": "/test/path", "file": "image2.jpg"},
    ]

def test_create_table(db_connection):
    """
    testet, ob die tabelle erfolgreich erstellt wurde
    """
    conn = db_connection
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
    assert c.fetchone() is not None, "tabelle wurde nicht erstellt"

def test_insert_and_count(db_connection, sample_image_data):
    """
    testet das einfügen von daten und das zählen der zeilen
    """
    conn = db_connection
    insert_data(conn, sample_image_data)
    
    # es wird geprüft, ob zwei zeilen eingefügt wurden
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM images")
    count = c.fetchone()[0]
    assert count == len(sample_image_data), f"erwartete {len(sample_image_data)} zeilen, aber fand stattdessen {count}"

def test_select_image_from_database(db_connection, sample_image_data):
    """
    zestet das abrufen von bildpfaden aus der datenbank
    """
    conn = db_connection
    insert_data(conn, sample_image_data)
    c = conn.cursor()

    file_path = select_image_from_database(1, c)
    assert file_path == "/test/path/image1.jpg", "derdDateipfad ist nicht korrekt"

    # testfall für ungültige image IDs
    assert select_image_from_database(999, c) is None, "nicht vorhandene ID sollte none zurückgeben"
