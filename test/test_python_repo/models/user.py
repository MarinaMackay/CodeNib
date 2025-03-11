# models/user.py - User model extending base
from models.base import Model, Credential
from utils.helpers import current_timestamp

class User(Model, Credential):
    def __init__(self, username, email=None):
        Model.__init__(self)
        Credential.__init__(self)
        self.username = username
        self.email = email
        self.is_active = True
        self.permissions = []
        self.created_at = current_timestamp()
        
    def save(self):
        self.updated_at = current_timestamp()
        return {"id": self.id, "username": self.username}
        
    def grant_permission(self, permission):
        if permission not in self.permissions:
            self.permissions.append(permission)
            
    def has_permission(self, permission):
        return permission in self.permissions

class Admin(User):
    def __init__(self, username, email=None):
        super().__init__(username, email)
        self.is_admin = True
        
    def grant_all_permissions(self):
        self.permissions = ["read", "write", "delete", "admin"]