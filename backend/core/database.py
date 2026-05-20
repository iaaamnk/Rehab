import sqlite3
import os
import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rehab_data.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_connection() as conn:
        c = conn.cursor()
        # Create Users Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        # Create Sessions Table
        c.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                rom_score REAL,
                stability_score REAL,
                quality_score REAL,
                total_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        conn.commit()

def create_user(username, password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed))
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        row = c.fetchone()
    if row:
        user_id, hashed = row
        if bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8')):
            return user_id
    return None

def log_session(user_id, date, rom, stability, quality, total):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO sessions (user_id, date, rom_score, stability_score, quality_score, total_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, date, rom, stability, quality, total))
        conn.commit()

def get_user_sessions(user_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            'SELECT date, rom_score, stability_score, quality_score, total_score '
            'FROM sessions WHERE user_id = ? ORDER BY id ASC',
            (user_id,)
        )
        rows = c.fetchall()
    return [
        {"date": r[0], "rom_score": r[1], "stability_score": r[2],
         "quality_score": r[3], "total_score": r[4]}
        for r in rows
    ]

# Initialize on import
init_db()

# To create an initial user, run:
#   from core.database import create_user
#   create_user('your_username', 'your_password')
