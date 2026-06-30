import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "detections.db"


def create_database():
    """
    Create the detections table if it doesn't exist.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            detected_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def save_detection(object_name, confidence):
    """
    Save one detected object to the database.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO detections
        (object_name, confidence, detected_at)
        VALUES (?, ?, ?)
        """,
        (
            object_name,
            confidence,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()


def load_detections():
    """
    Return all detections as a DataFrame.
    """

    conn = sqlite3.connect(DB_NAME)

    df = pd.read_sql_query(
        "SELECT * FROM detections ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


def clear_history():
    """
    Delete all detection history.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("DELETE FROM detections")

    conn.commit()
    conn.close()


def total_detections():
    """
    Return total number of detections.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM detections")

    total = cursor.fetchone()[0]

    conn.close()

    return total


def unique_objects():
    """
    Return number of unique object types.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(DISTINCT object_name) FROM detections"
    )

    total = cursor.fetchone()[0]

    conn.close()

    return total


def most_detected_object():
    """
    Return the most detected object.
    """

    conn = sqlite3.connect(DB_NAME)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT object_name
        FROM detections
        GROUP BY object_name
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """)

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return "No Data"