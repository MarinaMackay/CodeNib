"""
Similarity Service
Provides code similarity calculation and API client functionality
"""

from typing import Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
import time
import requests

from ..base import BaseAPI
from ...log_utils import get_logger

# Create logger for this module
logger = get_logger("api.services.similarity")

class SimilarityService:
    """
    Similarity service for code and natural language queries
    
    This service provides both:
    1. Model loading and similarity calculation logic (server-side)
    2. API client functionality for querying the service (client-side)
    """
    
    # Model configuration
    MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
    MAX_SEQ_LENGTH = 8192
    
    # Class variables
    _model = None
    
    @classmethod
    def preload_model(cls) -> None:
        """
        Preload the model during API server initialization
        This ensures the model is ready when the first request comes in
        """
        if cls._model is None:
            logger.info(f"Preloading model: {cls.MODEL_NAME}")
            cls._model = SentenceTransformer(
                cls.MODEL_NAME,
                trust_remote_code=True,
                device="cpu"  # Force using CPU
            )
            cls._model.max_seq_length = cls.MAX_SEQ_LENGTH
            logger.info("Model preloaded successfully")
    
    @classmethod
    def get_model(cls) -> SentenceTransformer:
        """Load and return the Sentence Transformer model (singleton)"""
        if cls._model is None:
            cls.preload_model()
        return cls._model
    
    @classmethod
    def calculate_similarity(cls, code: str, query: str) -> Dict[str, Any]:
        """
        Calculate similarity between code and query
        
        Args:
            code: Code text
            query: Query text
            
        Returns:
            dict: Dictionary containing score and latency
        """
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
        logger.debug(f"Similarity calculation completed in {latency}s, score: {score}")
        return {"score": score, "latency_sec": latency}
    
    @classmethod
    def query(cls, code: str, query: str) -> Dict[str, Any]:
        """
        Query the similarity between code and query (client method)
        
        This method sends a request to the API server to calculate similarity.
        Use this method when calling from other parts of your application.
        
        Args:
            code: Code text
            query: Query text
            
        Returns:
            dict: Dictionary containing score and latency
        """
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