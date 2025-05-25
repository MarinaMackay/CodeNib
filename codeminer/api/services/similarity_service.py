"""
Similarity Service
Provides code similarity calculation and API client functionality
"""

from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer, util
import time
import requests
import torch

from ..base import BaseAPI
from ...log_utils import get_logger

# Create logger for this module
logger = get_logger("api.services.similarity")

class SimilarityService:
    """
    Similarity service for code and natural language queries
    
    Simple and direct API for code similarity calculation with model and device switching
    """
    
    # Available embedding models
    AVAILABLE_MODELS = {
        "jina-embeddings-v2-base-code": {
            "name": "jinaai/jina-embeddings-v2-base-code",
            "max_seq_length": 8192,
        },
        "CodeRankEmbed": {
            "name": "nomic-ai/CodeRankEmbed", 
            "max_seq_length": 8192,
        }
    }
    
    # Default configuration
    DEFAULT_MODEL = "jina-embeddings-v2-base-code"
    DEFAULT_DEVICE = "cpu"  # auto, cpu, cuda, mps, etc.
    
    # Current configuration
    _current_model_key = DEFAULT_MODEL
    _current_device = DEFAULT_DEVICE
    
    # Class variables
    _model = None
    _model_cache = {}  # Cache for multiple models
    
    @classmethod
    def set_model(cls, model_key: str) -> None:
        """Set the embedding model to use"""
        if model_key not in cls.AVAILABLE_MODELS:
            available = list(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model '{model_key}' not available. Available models: {available}")
        
        if model_key != cls._current_model_key:
            cls._current_model_key = model_key
            cls._model = None  # Reset current model to force reload
            logger.info(f"Model set to: {model_key}")
    
    @classmethod
    def set_device(cls, device: str) -> None:
        """Set the compute device"""
        cls._current_device = device
        cls._model = None  # Reset current model to force reload with new device
        logger.info(f"Device set to: {device}")
    
    @classmethod
    def _get_actual_device(cls) -> str:
        """Resolve actual device from device setting"""
        if cls._current_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return cls._current_device
    
    @classmethod
    def _get_cache_key(cls) -> str:
        """Generate cache key for current model and device"""
        actual_device = cls._get_actual_device()
        return f"{cls._current_model_key}_{actual_device}"
    
    @classmethod
    def preload_model(cls) -> None:
        """Preload the model during API server initialization"""
        cache_key = cls._get_cache_key()
        
        if cache_key not in cls._model_cache:
            model_config = cls.AVAILABLE_MODELS[cls._current_model_key]
            actual_device = cls._get_actual_device()
            
            logger.info(f"Preloading model: {model_config['name']} on device: {actual_device}")
            
            model = SentenceTransformer(
                model_config["name"],
                trust_remote_code=True,
                device=actual_device
            )
            model.max_seq_length = model_config["max_seq_length"]
            
            cls._model_cache[cache_key] = model
            logger.info(f"Model preloaded successfully: {cache_key}")
        
        cls._model = cls._model_cache[cache_key]
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Load and return the Sentence Transformer model (with caching)"""
        if cls._model is None:
            cls.preload_model()
        return cls._model
    
    @classmethod
    def calculate_similarity(cls, code: str, query: str) -> Dict[str, Any]:
        """Calculate similarity between code and query"""
        start_time = time.time()
        
        # Get model
        model = cls.get_model()
        
        # Encode and calculate similarity
        vec_q, vec_c = model.encode(
            [query, code],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        score = float(util.cos_sim(vec_q, vec_c).item())
        
        latency = round(time.time() - start_time, 4)
        
        result = {
            "score": score, 
            "latency_sec": latency
        }
        
        logger.debug(f"Similarity calculation completed in {latency}s, score: {score}")
        return result
    
    @classmethod
    def query(cls, code: str, query: str) -> Dict[str, Any]:
        """Query the similarity between code and query (client method)"""
        # Get host and port from BaseAPI
        host = BaseAPI._host
        port = BaseAPI._port
        
        if not BaseAPI._initialized:
            logger.error("API server is not started, cannot query similarity")
            raise RuntimeError("API server is not started, please use BaseAPI.start() first")
        
        url = f"http://{host}:{port}/similarity"
        payload = {"code": code, "query": query}
        
        logger.debug(f"Sending similarity request to {url}")
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
    
    @classmethod
    def configure(cls, model: Optional[str] = None, device: Optional[str] = None) -> Dict[str, Any]:
        """Configure model and device (client method)"""
        # Get host and port from BaseAPI
        host = BaseAPI._host
        port = BaseAPI._port
        
        if not BaseAPI._initialized:
            logger.error("API server is not started, cannot configure")
            raise RuntimeError("API server is not started, please use BaseAPI.start() first")
        
        url = f"http://{host}:{port}/similarity/configure"
        payload = {}
        if model is not None:
            payload["model"] = model
        if device is not None:
            payload["device"] = device
        
        logger.debug(f"Sending configuration request to {url}")
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"} 