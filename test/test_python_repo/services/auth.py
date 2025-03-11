# services/auth.py - Authentication service
from utils.helpers import generate_token
from models.user import User

class AuthResult:
    def __init__(self, success, user=None, token=None):
        self.success = success
        self.user = user
        self.token = token

class AuthService:
    def __init__(self, db_manager):
        self.db = db_manager
        self.active_sessions = {}
        
    def verify_credentials(self, username, password):
        user_data = self.db.find_user(username)
        if not user_data:
            return AuthResult(False)
            
        # Create user from data
        user = User(user_data["username"])
        user.id = user_data["id"]
        
        if not user_data.get("password_hash"):
            return AuthResult(False)
            
        # Set stored credentials
        user.password_hash = user_data["password_hash"]
        user.salt = user_data["salt"]
        
        # Verify password
        if user.verify_password(password):
            token = generate_token()
            self.active_sessions[token] = user.id
            return AuthResult(True, user, token)
        
        return AuthResult(False)
        
    def logout(self, token):
        if token in self.active_sessions:
            del self.active_sessions[token]
            return True
        return False