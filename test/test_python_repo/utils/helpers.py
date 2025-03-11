# utils/helpers.py - Helper functions
import time
import uuid
import json
import os
from datetime import datetime

def generate_id():
    """Generate a unique ID"""
    return str(uuid.uuid4())

def current_timestamp():
    """Return current timestamp"""
    return datetime.now().isoformat()

def generate_token():
    """Generate a secure token"""
    return generate_id() + str(int(time.time()))

def load_config(path="config.json"):
    """Load configuration from file"""
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"db_path": ":memory:", "debug": True}

def format_date(date_string, format="%Y-%m-%d"):
    """Format a date string"""
    dt = datetime.fromisoformat(date_string)
    return dt.strftime(format)