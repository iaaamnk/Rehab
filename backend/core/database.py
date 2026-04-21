import sqlite3
import os
import bcrypt

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rehab_data.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
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
            posture_score REAL,
            total_score REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password):
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        user_id, hashed = row
        if bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8')):
            return user_id
    return None

def log_session(user_id, date, rom, stability, posture, total):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO sessions (user_id, date, rom_score, stability_score, posture_score, total_score)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, date, rom, stability, posture, total))
    conn.commit()
    conn.close()

def get_user_sessions(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute('SELECT date, rom_score, stability_score, posture_score, total_score FROM sessions WHERE user_id = ? ORDER BY id ASC', (user_id,))
    rows = c.fetchall()
    conn.close()
    
    sessions = []
    for r in rows:
         sessions.append({
             "date": r[0],
             "rom_score": r[1],
             "stability_score": r[2],
             "posture_score": r[3],
             "total_score": r[4]
         })
    return sessions

# Initialize on import
init_db()

# To create an initial user, run:
#   from core.database import create_user
#   create_user('your_username', 'your_password')
