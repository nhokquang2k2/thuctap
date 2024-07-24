import sqlite3
import os
import numpy as np

def get_connection():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, '../data', 'data.db')
    return sqlite3.connect(db_path)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, role TEXT, approved INTEGER, image BLOB, encoding BLOB)''')
    conn.commit()
    update_db_structure()
    conn.close()

def update_db_structure():
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN last_attendance_time TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def get_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT username, role, approved, last_attendance_time FROM users")
    users = c.fetchall()
    conn.close()
    return users

def get_unapproved_users():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT username, role FROM users WHERE approved = 0")
    users = c.fetchall()
    conn.close()
    return users

def approve_user(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET approved = 1 WHERE username = ?", (username,))
    conn.commit()
    conn.close()

def register_user(username, password, role, image_path):
    conn = get_connection()
    c = conn.cursor()
    
    with open(image_path, 'rb') as file:
        image_data = file.read()
    
    c.execute("INSERT INTO users (username, password, role, approved, image) VALUES (?, ?, ?, ?, ?)",
              (username, password, role, 0, image_data))
    conn.commit()
    conn.close()

def authenticate_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ? AND approved = 1", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def save_attendance(username, time):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET last_attendance_time = ? WHERE username = ?", (time, username))
    conn.commit()
    conn.close()

def get_attendance_records():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT username, last_attendance_time FROM users WHERE last_attendance_time IS NOT NULL")
    records = c.fetchall()
    conn.close()
    return records

def get_user_image(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT image FROM users WHERE username = ?", (username,))
    image_data = c.fetchone()
    conn.close()
    return image_data[0] if image_data else None

def delete_user(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()
    conn.close()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
    npy_file = os.path.join(dataset_path, f'{username}.npy')
    if os.path.exists(npy_file):
        os.remove(npy_file)

def user_exists(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username,))
    exists = c.fetchone()[0] > 0
    conn.close()
    return exists

def get_all_face_encodings():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT username, encoding FROM users WHERE approved = 1")
    users = c.fetchall()
    conn.close()
    encodings = []
    usernames = []
    for user in users:
        encodings.append(np.frombuffer(user[1], dtype=np.float64))
        usernames.append(user[0])
    return usernames, encodings
