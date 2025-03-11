# models/base.py - Base models
import hashlib
from utils.helpers import generate_id

class Model:
    def __init__(self, id=None):
        self.id = id or generate_id()
        self.created_at = None
        self.updated_at = None
    
    def save(self):
        # This will be overridden by subclasses
        pass
        
    def validate(self):
        return bool(self.id)

class Credential:
    def __init__(self):
        self.password_hash = None
        self.salt = None
        
    def set_password(self, password):
        self.salt = generate_id()[:8]
        self.password_hash = hashlib.sha256((password + self.salt).encode()).hexdigest()
        
    def verify_password(self, password):
        hash_to_check = hashlib.sha256((password + self.salt).encode()).hexdigest()
        return hash_to_check == self.password_hash