"""
SQLite Database for AFLCP Application
Handles user authentication and session management
"""
import sqlite3
import hashlib
import secrets
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aflcp.db")


def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT DEFAULT 'researcher',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            model_name TEXT,
            config TEXT,
            final_accuracy REAL,
            final_f1 REAL,
            final_auc REAL,
            total_rounds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create default users if not exist
    default_users = [
        ("admin", "admin123", "Administrator", "admin"),
        ("doctor", "doctor123", "Dr. Sarah Chen", "researcher"),
    ]
    for username, password, full_name, role in default_users:
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if not cursor.fetchone():
            salt = secrets.token_hex(16)
            password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, full_name, role) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, full_name, role)
            )

    conn.commit()
    conn.close()


def hash_password(password, salt=None):
    """Hash a password with salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt


def create_user(username, password, full_name, role="researcher"):
    """Create a new user"""
    conn = get_db()
    cursor = conn.cursor()
    password_hash, salt = hash_password(password)
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, salt, full_name, role) VALUES (?, ?, ?, ?, ?)",
            (username, password_hash, salt, full_name, role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    """Verify user credentials and return user dict"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return None

    password_hash = hashlib.sha256((password + user["salt"]).encode()).hexdigest()
    if password_hash == user["password_hash"]:
        return dict(user)
    return None


def create_session(user_id):
    """Create a new session token"""
    conn = get_db()
    cursor = conn.cursor()
    token = secrets.token_hex(32)
    cursor.execute(
        "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
        (token, user_id)
    )
    conn.commit()
    conn.close()
    return token


def get_user_from_session(token):
    """Get user from session token"""
    if not token:
        return None
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.* FROM users u
        JOIN sessions s ON u.id = s.user_id
        WHERE s.token = ?
    """, (token,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None


def delete_session(token):
    """Delete a session"""
    if not token:
        return
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def save_training_record(user_id, model_name, config, accuracy, f1, auc, rounds):
    """Save training history record"""
    conn = get_db()
    cursor = conn.cursor()
    import json
    cursor.execute(
        """INSERT INTO training_history
           (user_id, model_name, config, final_accuracy, final_f1, final_auc, total_rounds)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (user_id, model_name, json.dumps(config) if config else None, accuracy, f1, auc, rounds)
    )
    conn.commit()
    conn.close()


# Initialize database on import
init_db()
