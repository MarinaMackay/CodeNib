"""
Base API Framework
Provides the core API server with registration and management capabilities
"""

from typing import AsyncGenerator, Dict, Any, Optional, List, Type
from fastapi import FastAPI, APIRouter
import time
import uvicorn
import threading
import logging
import atexit
import signal
import sys
import inspect
from contextlib import asynccontextmanager
import asyncio

from ..log_utils import get_logger

# Create logger for this module
logger = get_logger("api.base")

class BaseAPI:
    """Base API framework providing registration and automatic management functionality"""
    
    # Default configuration
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000
    
    # Class variables
    _instance = None
    _api_server = None
    _host = DEFAULT_HOST
    _port = DEFAULT_PORT
    _initialized = False
    _log_level = "warning"
    _uvicorn_server = None
    _exit_event = None
    _app = None
    _routers = []
    _api_registry = []  # Store all API classes
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(BaseAPI, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def _create_api_app(cls) -> FastAPI:
        """Create FastAPI application instance"""
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            try:
                # Perform initialization operations
                logger.info("Initializing API services...")
                
                # Initialize API services that need pre-loading
                # This is a good place to preload models, establish connections, etc.
                from .services.similarity_service import SimilarityService
                SimilarityService.preload_model()
                
                yield
                
                # Cleanup operations when shutting down
                logger.info("Cleaning up resources...")
            except asyncio.CancelledError:
                # Gracefully handle cancellation during shutdown
                logger.debug("Lifespan context was cancelled during shutdown - this is normal")
            except Exception as e:
                logger.error(f"Error in lifespan context: {str(e)}")
                raise
        
        app = FastAPI(
            title="CodeMiner API",
            description="CodeMiner API framework",
            version="0.1",
            lifespan=lifespan
        )
        
        # Register all routes
        for router in cls._routers:
            app.include_router(router)
            
        return app
    
    @classmethod
    def register_router(cls, router: APIRouter) -> None:
        """
        Register new API routes

        Args:
            router: FastAPI route object
        """
        cls._routers.append(router)
        # If application has been created, dynamically add routes
        if cls._app is not None:
            cls._app.include_router(router)
    
    @classmethod
    def register_api(cls, api_class: Type) -> None:
        """
        Register API class
        
        Args:
            api_class: API class to register with the framework
        """
        if api_class not in cls._api_registry:
            cls._api_registry.append(api_class)
    
    @classmethod
    def _setup_exit_handlers(cls):
        """Set shutdown handlers for program exit"""
        def exit_handler():
            if cls._initialized:
                cls.stop()
        
        # Register exit handler
        atexit.register(exit_handler)
        
        # Set signal handler
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down API server...")
            cls.stop()
            sys.exit(0)
        
        # Register common termination signal handlers
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Terminate request

    @classmethod
    def _register_default_routes(cls):
        """Register default API routes"""
        # Import routers here to avoid circular import
        from .routes.similarity_routes import router as similarity_router
        
        # Register routers
        cls.register_router(similarity_router)
        
        # Add new API routers here as they are created
        # For example:
        # from .routes.new_feature_routes import router as new_feature_router
        # cls.register_router(new_feature_router)

    @classmethod
    def start(cls, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str = "warning") -> None:
        """
        Start API server
        
        Args:
            host: API server host address, default is 127.0.0.1
            port: API server port, default is 8000
            log_level: Uvicorn log level: ["critical", "error", "warning", "info", "debug", "trace"]
                     default is "warning", only show warning and error
        """
        # Save configuration
        cls._host = host
        cls._port = port
        cls._log_level = log_level.lower()
        
        # Validate log level
        valid_levels = ["critical", "error", "warning", "info", "debug", "trace"]
        if cls._log_level not in valid_levels:
            logger.warning(f"Invalid log level '{log_level}', using default value 'warning'")
            cls._log_level = "warning"
            
        # If server is already running, return directly
        if cls._api_server is not None and cls._api_server.is_alive():
            logger.info(f"API server is already running at http://{cls._host}:{cls._port}")
            return
        
        # Register default routes
        cls._register_default_routes()
        
        # Initialize all registered APIs
        for api_class in cls._api_registry:
            if hasattr(api_class, 'init'):
                api_class.init()
        
        # Create API application
        cls._app = cls._create_api_app()
        
        # Create thread event for server shutdown
        cls._exit_event = threading.Event()
        
        # Start server in background thread
        def run_api():
            # Configure uvicorn logging
            cls._uvicorn_server = uvicorn.Server(uvicorn.Config(
                cls._app,
                host=cls._host,
                port=cls._port,
                log_level=cls._log_level
            ))
            cls._uvicorn_server.run()
        
        cls._api_server = threading.Thread(target=run_api, daemon=True)
        cls._api_server.start()
        cls._initialized = True
        time.sleep(1)
        logger.info(f"API server started at http://{cls._host}:{cls._port}")
        
        # Set exit handler
        cls._setup_exit_handlers()
    
    @classmethod
    def stop(cls) -> None:
        """
        Stop API server
        
        This method should be called when the program is about to exit,
        to ensure resources are properly cleaned up.
        """
        if not cls._initialized:
            return
            
        logger.info("Stopping API server...")
        
        # Check if server is running before stopping
        if not cls._api_server or not cls._api_server.is_alive():
            logger.info("API server is not running")
            return
            
        # Notify uvicorn server to close
        if cls._uvicorn_server:
            cls._uvicorn_server.should_exit = True
            
            # Give lifespan tasks time to complete properly
            time.sleep(0.5)
            
        # Wait for server thread to terminate (with timeout)
        if cls._api_server and cls._api_server.is_alive():
            cls._api_server.join(timeout=2.0)
            
            # If server is still alive after timeout, force it to stop
            if cls._api_server.is_alive():
                logger.warning("Server did not stop gracefully, forcing shutdown")
                # In production code, you might want to avoid force=True
                # but for testing it's acceptable
                cls._uvicorn_server.force_exit = True
                cls._api_server.join(timeout=1.0)
        
        # Reset status
        cls._initialized = False
        cls._api_server = None
        cls._uvicorn_server = None
        logger.info("API server has been stopped")
    
    @classmethod
    def auto_start_on_import(cls):
        """
        Decorator to automatically start the API server
        Used to automatically start the API server when the module is imported
        """
        # Detect call stack, determine if it is in the main module
        try:
            for frame in inspect.stack():
                if '__main__' in sys.modules and hasattr(sys.modules['__main__'], '__file__'):
                    if frame.filename == sys.modules['__main__'].__file__:
                        cls.start()
                        break
        except (AttributeError, KeyError):
            # In some environments like interactive shells, __main__ might not have __file__
            # In that case, don't auto-start
            pass
        return cls 