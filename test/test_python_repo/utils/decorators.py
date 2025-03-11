# utils/decorators.py - Function decorators
import time
import functools
from utils.helpers import current_timestamp

def timing(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def logging(func):
    """Decorator to log function calls"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"[{current_timestamp()}] Calling {func.__name__}")
        try:
            result = func(*args, **kwargs)
            print(f"[{current_timestamp()}] {func.__name__} completed successfully")
            return result
        except Exception as e:
            print(f"[{current_timestamp()}] {func.__name__} failed with error: {str(e)}")
            raise
    return wrapper

def require_auth(func):
    """Decorator to require authentication"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'is_authenticated') or not self.is_authenticated:
            raise ValueError("Authentication required")
        return func(self, *args, **kwargs)
    return wrapper