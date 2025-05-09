from functools import lru_cache
from typing import AsyncGenerator, Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI
from pydantic import BaseModel
import time
import requests
from contextlib import asynccontextmanager
import uvicorn
import threading
import logging
import atexit
import signal
import sys

class SimilarityAPI:
    """Code similarity API wrapper class, providing model loading and HTTP request functionality"""
    
    # Default configuration
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000
    MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
    MAX_SEQ_LENGTH = 8192
    
    # Class variables
    _instance = None
    _model = None
    _api_server = None
    _host = DEFAULT_HOST
    _port = DEFAULT_PORT
    _initialized = False
    _log_level = "warning"
    _uvicorn_server = None
    _exit_event = None
    
    def __new__(cls):
        """Implement the singleton pattern"""
        if cls._instance is None:
            cls._instance = super(SimilarityAPI, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        """Load the Sentence Transformer model"""
        model = SentenceTransformer(
            cls.MODEL_NAME,
            trust_remote_code=True,
            device="cpu"  # Force using CPU
        )
        model.max_seq_length = cls.MAX_SEQ_LENGTH
        return model
    
    @classmethod
    def _create_api_app(cls) -> FastAPI:
        """Create a FastAPI application instance"""
        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            # Warm up the model
            if cls._model is None:
                cls._model = cls._load_model()
            yield
        
        app = FastAPI(
            title="Code–Query Similarity API",
            description="Calculate cosine similarity between code and query using Jina embeddings.",
            version="0.1",
            lifespan=lifespan
        )

        class SimRequest(BaseModel):
            code: str
            query: str

        @app.post("/similarity")
        def similarity(req: SimRequest):
            try:
                # Ensure the model is loaded
                if cls._model is None:
                    cls._model = cls._load_model()
                
                start = time.time()
                
                # Encode and calculate similarity
                vec_q, vec_c = cls._model.encode(
                    [req.query, req.code],
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                score = float(util.cos_sim(vec_q, vec_c).item())
                
                latency = round(time.time() - start, 4)
                return {"score": score, "latency_sec": latency}
            except Exception as e:
                return {"error": str(e)}
        
        return app
    
    @classmethod
    def _setup_exit_handlers(cls):
        """Setup handlers for graceful shutdown when program exits"""
        def exit_handler():
            if cls._initialized:
                cls.stop()
        
        # Register the exit handler to be called when program exits
        atexit.register(exit_handler)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            print(f"\nReceived signal {sig}, shutting down API server...")
            cls.stop()
            sys.exit(0)
        
        # Register signal handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # termination request
    
    @classmethod
    def start(cls, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, log_level: str = "warning") -> None:
        """
        Start the similarity API server
        
        Args:
            host: API server host address, default is 127.0.0.1
            port: API server port, default is 8000
            log_level: Uvicorn log level, optional values are ["critical", "error", "warning", "info", "debug", "trace"]
                      default is "warning", only shows warnings and errors
        """
        # Save the configuration
        cls._host = host
        cls._port = port
        cls._log_level = log_level.lower()
        
        # Validate the log level
        valid_levels = ["critical", "error", "warning", "info", "debug", "trace"]
        if cls._log_level not in valid_levels:
            print(f"Warning: invalid log level '{log_level}', using default value 'warning'")
            cls._log_level = "warning"
            
        # If the service is already running, return directly
        if cls._api_server is not None and cls._api_server.is_alive():
            print(f"API server is already running at http://{cls._host}:{cls._port}")
            return
        
        # Preload the model
        if cls._model is None:
            cls._model = cls._load_model()
        
        # Create API application
        api_app = cls._create_api_app()
        
        # Create threading event for server shutdown
        cls._exit_event = threading.Event()
        
        # Start the service in a background thread
        def run_api():
            # Configure uvicorn logging
            cls._uvicorn_server = uvicorn.Server(uvicorn.Config(
                api_app,
                host=cls._host,
                port=cls._port,
                log_level=cls._log_level
            ))
            cls._uvicorn_server.run()
        
        cls._api_server = threading.Thread(target=run_api, daemon=True)
        cls._api_server.start()
        cls._initialized = True
        time.sleep(1)
        print(f"Similarity API server started at http://{cls._host}:{cls._port}")
        
        # Setup exit handlers for automatic cleanup
        cls._setup_exit_handlers()
    
    @classmethod
    def stop(cls) -> None:
        """
        Stop the API server
        
        This method should be called when the program is about to exit
        to ensure proper cleanup of resources.
        """
        if not cls._initialized:
            return
            
        print("Stopping Similarity API server...")
        
        # Signal the uvicorn server to shut down
        if cls._uvicorn_server:
            cls._uvicorn_server.should_exit = True
            
        # Wait for the server thread to terminate (with timeout)
        if cls._api_server and cls._api_server.is_alive():
            cls._api_server.join(timeout=2.0)
        
        # Reset state
        cls._initialized = False
        cls._api_server = None
        cls._uvicorn_server = None
        print("Similarity API server stopped")
    
    @classmethod
    def query(cls, code: str, query: str) -> Dict[str, Any]:
        """
        Calculate the similarity between code and query
        
        Args:
            code: code text
            query: query text
            
        Returns:
            dict: contains the score and latency
        """
        if not cls._initialized:
            raise RuntimeError("API Server is not started, please start the server first.")
        
        url = f"http://{cls._host}:{cls._port}/similarity"
        payload = {"code": code, "query": query}
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}