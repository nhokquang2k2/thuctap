import sqlite3
import os

DATABASE_PATH = os.path.join('data', 'data.db')

def add_admin():
    with sqlite3.connect(DATABASE_PATH) as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password, role, approved, image) VALUES (?, ?, ?, ?, ?)",
                      ('admin', '123123', 'admin', 1, None))
            conn.commit()
        except sqlite3.IntegrityError:
            print("Admin already exists.")

if __name__ == '__main__':
    add_admin()
