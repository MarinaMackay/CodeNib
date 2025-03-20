# services/data/db.py - Database access
from utils.helpers import current_timestamp


class QueryResult:
    def __init__(self, success, data=None, error=None):
        self.success = success
        self.data = data or []
        self.error = error


class Cursor:
    def __init__(self, connection):
        self.connection = connection

    def execute(self, query):
        print(f"Executing: {query}")
        return QueryResult(True, [{"id": 1, "name": "test"}])


class Connection:
    def __init__(self, db_path):
        self.db_path = db_path
        self.cursor = Cursor(self)
        self.is_connected = False

    def connect(self):
        self.is_connected = True
        current_timestamp()

    def close(self):
        self.is_connected = False


class DatabaseManager:
    def __init__(self, config=None):
        self.config = config or {"db_path": ":memory:"}
        self.connection = Connection(self.config.get("db_path"))

    def connect(self):
        self.connection.connect()
        return self

    def execute_query(self, query):
        if not self.connection.is_connected:
            self.connection.connect()
        return self.connection.cursor.execute(query)

    def save_user(self, user):
        # Simulate saving user to database
        query = f"INSERT INTO users (username, password_hash) VALUES ('{user.username}', '{user.password_hash}')"
        result = self.execute_query(query)
        return result

    def find_user(self, username):
        query = f"SELECT * FROM users WHERE username='{username}'"
        result = self.execute_query(query)

        if result.success and len(result.data) > 0:
            user_data = result.data[0]
            user_data["password_hash"] = "dummy_hash"
            user_data["salt"] = "dummy_salt"
            return user_data
        return None
