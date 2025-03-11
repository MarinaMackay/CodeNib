# main.py - Entry point with complex import patterns
from models.user import User, Admin
from services.auth import AuthService
from services.data.db import DatabaseManager
import utils.helpers as helpers
from utils.decorators import timing, logging

class Application:
    def __init__(self, config=None):
        self.db = DatabaseManager(config)
        self.auth_service = AuthService(self.db)
        self.users = {}
        
    def register_user(self, username, password):
        # Direct method call
        user = User(username)
        user.set_password(password)
        
        # Method call on object attribute
        result = self.db.save_user(user)
        
        # Access attribute of imported class instance
        if result.success:
            self.users[username] = user
            return True
        return False

    def authenticate(self, username, password):
        # Call to nested attribute
        return self.auth_service.verify_credentials(username, password)
        
    @timing
    @logging
    def run(self):
        # Call to imported function as helper
        config = helpers.load_config()
        
        # Instantiate class from another module
        admin = Admin("admin")
        admin.set_password("admin123")
        admin.grant_all_permissions()
        
        # Method chaining across modules
        self.db.connect().execute_query("SELECT * FROM users")
        
        # Nested attribute across different modules
        self.db.connection.cursor.execute("SELECT * FROM users")
        
        print("Application running...")

if __name__ == "__main__":
    app = Application()
    app.register_user("test", "password")
    app.authenticate("test", "password")
    app.run()